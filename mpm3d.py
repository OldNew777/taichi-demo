import shutil
import os

import taichi as ti
import numpy as np
import mcubes
import scipy.spatial

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2
import tqdm


@ti.func
def gaussian_random(sigma: float = 1.0, mu: float = 0.0):
    u1 = ti.random(dtype=float)
    u2 = ti.random(dtype=float)
    return sigma * ti.sqrt(-2 * ti.log(u1)) * ti.cos(2 * np.pi * u2) + mu


class ParticleStatus(int):
    INVALID = 0
    VALID = 1


@ti.data_oriented
class MPM3DSimulator:
    def __init__(self):
        ti.init(arch=ti.cuda)

        self.n_steps = 50
        self.n_grids = 64
        self.n_insert_per_step = 64
        self.radius_insert_last = ti.field(dtype=float, shape=(1,))
        self.radius_insert = 0.03
        self.radius_insert_delta = 0.05
        self.pos_insert = ti.Vector([0.2, 0.95, 0.7])
        self.v_init_insert = ti.Vector([1.5, -5.0, -1.2])

        self.n_particles = (self.n_grids ** 3) // (2 ** 3)
        self.n_valid = self.n_particles

        # constant physical values for simulation
        self.dx = 1. / self.n_grids
        self.dt = 8e-5
        self.p_rho = 1.
        self.p_vol = (self.dx * 0.5) ** 3
        self.p_mass = self.p_vol * self.p_rho
        self.gravity = 9.8
        self.bound = 3
        self.E = 400.

        # values for record
        self.x = ti.Vector.field(n=3, dtype=float, shape=(self.n_particles,))
        self.v = ti.Vector.field(n=3, dtype=float, shape=(self.n_particles,))
        self.J = ti.field(dtype=float, shape=(self.n_particles,))
        self.C = ti.Matrix.field(n=3, m=3, dtype=float, shape=(self.n_particles,))
        self.status = ti.field(dtype=int, shape=(self.n_particles,))
        self.grid_v = ti.Vector.field(n=3, dtype=float, shape=(self.n_grids, self.n_grids, self.n_grids))
        self.grid_m = ti.field(dtype=float, shape=(self.n_grids, self.n_grids, self.n_grids))

        # attribute cache
        self.insert_index = self.n_particles - 1
        self.insert_indexes = ti.field(dtype=int, shape=(self.n_insert_per_step,))

        # rendering parameters
        self.save_frames = True
        self.save_obj = False
        self.save_seconds = 10.0
        self.frame_rate = 1. / self.dt / self.n_steps
        self.resolution = (1024, 1024)
        self.background_color = (0.2, 0.2, 0.4)
        self.point_color = (0.4, 0.6, 0.6)
        self.point_radius = 0.003
        self.reconstruct_threshold = 0.75
        self.reconstruct_resolution = (100, 100, 100)
        self.reconstruct_radius = 0.1

        self.bounding_box_vertices = ti.Vector.field(n=3, dtype=float, shape=(24,))

    @ti.func
    def get_grid_index(self, xyz):
        return ti.math.clamp(ti.cast(xyz, ti.i32), 0, self.n_grids - 1)

    @ti.kernel
    def initialize(self):
        for index in ti.grouped(self.x):
            self.x[index] = [ti.random() * 0.4 + 0.2, ti.random() * 0.4 + 0.2, ti.random() * 0.4 + 0.2]
        self.v.fill([0., 0., 0.])
        self.J.fill(1.)
        self.C.fill(0.)
        self.status.fill(ParticleStatus.VALID)
        self.radius_insert_last[0] = 0.

        bound_exact = (self.bound - 0.7) / self.n_grids
        pos = [bound_exact, 1. - bound_exact]
        for axis, i, j, k in ti.static(ti.ndrange(3, 2, 2, 2)):
            index = axis * 8 + i * 4 + j * 2 + k
            self.bounding_box_vertices[index][(axis + 0) % 3] = pos[i]
            self.bounding_box_vertices[index][(axis + 1) % 3] = pos[j]
            self.bounding_box_vertices[index][(axis + 2) % 3] = pos[k]


    @ti.kernel
    def clear_grid(self):
        self.grid_v.fill([0., 0., 0.])
        self.grid_m.fill(0.)

    def reverse(self, size_new: int):
        assert size_new >= self.n_particles
        if size_new == self.n_particles:
            return

        self.x_new = ti.Vector.field(n=3, dtype=float, shape=(size_new,))
        self.v_new = ti.Vector.field(n=3, dtype=float, shape=(size_new,))
        self.J_new = ti.field(dtype=float, shape=(size_new,))
        self.C_new = ti.Matrix.field(n=3, m=3, dtype=float, shape=(size_new,))
        self.status_new = ti.field(dtype=int, shape=(size_new,))

        self.x_new.fill([0., 0., 0.])
        self.v_new.fill([0., 0., 0.])
        self.J_new.fill(1.)
        self.C_new.fill(0.)
        self.status_new.fill(ParticleStatus.INVALID)

        self.copy_from_new()

        self.x = self.x_new
        self.v = self.v_new
        self.J = self.J_new
        self.C = self.C_new
        self.status = self.status_new
        del self.x_new
        del self.v_new
        del self.J_new
        del self.C_new
        del self.status_new

        self.n_particles = size_new

    @ti.kernel
    def copy_from_new(self):
        for index in ti.grouped(self.x):
            if self.status[index] == ParticleStatus.VALID:
                self.x_new[index] = self.x[index]
                self.v_new[index] = self.v[index]
                self.J_new[index] = self.J[index]
                self.C_new[index] = self.C[index]
                self.status_new[index] = self.status[index]

    # @ti.kernel
    # def valid_n(self) -> int:
    #     n = ti.int32(0)
    #     for i in ti.grouped(self.status):
    #         if self.status[i] == ParticleStatus.VALID:
    #             ti.atomic_add(n, 1)
    #     return int(n)

    def valid_n(self) -> int:
        return self.n_valid

    @ti.kernel
    def init_added_particles(self):
        for i in ti.grouped(self.insert_indexes):
            index = self.insert_indexes[i]
            theta = ti.random() * 2 * np.pi
            x_offset = (ti.random() - 0.5) * self.v_init_insert[1] * self.dt * self.n_steps
            r = 10000000.
            while ti.abs(r - self.radius_insert_last[0]) > self.radius_insert_delta or r > self.radius_insert:
                r = ti.abs(gaussian_random()) * self.radius_insert
            self.radius_insert_last[0] = r
            self.x[index] = [r * ti.cos(theta), 0., r * ti.sin(theta)] + self.pos_insert + x_offset
            self.v[index] = self.v_init_insert
            self.J[index] = 1.
            self.C[index] = ti.Matrix.zero(float, 3, 3)
            self.status[index] = ParticleStatus.VALID

    def add_particles(self):
        n = self.n_insert_per_step
        if self.valid_n() + n > self.n_particles:
            new_size = self.n_particles + max(n, self.n_particles, 1 << 20)
            print(f'Reverse from {self.n_particles} to {new_size}', )
            self.reverse(new_size)
        insert_indexes_np = []
        while n > 0:
            while self.status[self.insert_index] != ParticleStatus.INVALID or self.insert_index in insert_indexes_np:
                self.insert_index = (self.insert_index + 1) % self.n_particles
            n -= 1
            insert_indexes_np.append(self.insert_index)
        insert_indexes_np = np.array(insert_indexes_np)
        self.insert_indexes.from_numpy(insert_indexes_np)
        self.init_added_particles()
        self.n_valid += self.n_insert_per_step

    @ti.func
    def weight_kernel(self, x):
        Xp = x / self.dx
        base = ti.cast(Xp - 0.5, ti.i32)
        fx = Xp - base.cast(float)
        w_l = 0.5 * ti.pow((1.5 - fx), 2)
        w_c = 0.75 - ti.pow((fx - 1), 2)
        w_r = 0.5 * ti.pow((fx - 0.5), 2)
        w = ti.Matrix.rows([w_l, w_c, w_r])
        return base, fx, w

    @ti.kernel
    def point_to_grid(self):
        for index in ti.grouped(self.x):
            if self.status[index] == ParticleStatus.VALID:
                base, fx, w = self.weight_kernel(self.x[index])
                stress = -4. * self.dt * self.E * self.p_vol * (self.J[index] - 1.) / ti.pow(self.dx, 2)
                affine = ti.Matrix.diag(3, stress) + self.p_mass * self.C[index]
                for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                    offset = ti.Vector([i, j, k], dt=int)
                    dpos = (offset.cast(float) - fx) * self.dx
                    weight = w[i, 0] * w[j, 1] * w[k, 2]
                    v_add = weight * (self.p_mass * self.v[index] + affine @ dpos)
                    gird_index_new = self.get_grid_index(base + offset)
                    ti.atomic_add(self.grid_v[gird_index_new], v_add)
                    ti.atomic_add(self.grid_m[gird_index_new], weight * self.p_mass)

    @ti.kernel
    def simulate_grid(self):
        for i, j, k in ti.ndrange(self.grid_v.shape[0], self.grid_v.shape[1], self.grid_v.shape[2]):
            m_temp = self.grid_m[i, j, k]
            v_temp = ti.select(m_temp > 0, self.grid_v[i, j, k] / m_temp, ti.Vector.zero(float, 3))
            v_temp[1] -= self.dt * self.gravity
            index = ti.Vector([i, j, k], dt=int)
            cond = (index < self.bound and v_temp < 0.) or (index > self.n_grids - self.bound and v_temp > 0.)
            v_temp = ti.select(cond, 0., v_temp)
            self.grid_v[i, j, k] = v_temp

    @ti.kernel
    def grid_to_point(self):
        for index in ti.grouped(self.x):
            if self.status[index] == ParticleStatus.VALID:
                base, fx, w = self.weight_kernel(self.x[index])
                new_v = ti.Vector.zero(float, 3)
                new_C = ti.Matrix.zero(float, 3, 3)
                for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                    offset = ti.Vector([i, j, k], dt=int)
                    dpos = (offset.cast(float) - fx) * self.dx
                    weight = w[i, 0] * w[j, 1] * w[k, 2]
                    gird_index_new = self.get_grid_index(base + offset)
                    g_v = self.grid_v[gird_index_new]
                    new_v += weight * g_v
                    new_C += 4. * weight * g_v.outer_product(dpos) / ti.pow(self.dx, 2)
                self.v[index] = new_v
                self.x[index] += self.dt * new_v
                self.J[index] *= 1. + self.dt * new_C.trace()
                self.C[index] = new_C

    def substep(self):
        self.clear_grid()
        self.point_to_grid()
        self.simulate_grid()
        self.grid_to_point()

    def step(self):
        self.add_particles()
        for i in range(self.n_steps):
            self.substep()

    def metaball_scalar_field(self, resolution):
        dx = self.dx / resolution[0]
        dy = self.dx / resolution[1]
        dz = self.dx / resolution[2]

        radius = self.reconstruct_radius

        particle_pos = self.x.to_numpy()
        kdtree = scipy.spatial.KDTree(particle_pos)

        # We are picking scalar values at vertices of a cell rather than the center
        # Also, step out one cell's distance to completely include the simulation region
        f_shape = (resolution[0] + 3, resolution[1] + 3, resolution[2] + 3)
        i, j, k = np.mgrid[:f_shape[0], :f_shape[1], :f_shape[2]] - 1

        x = i * dx
        y = j * dy
        z = k * dz
        field_pos = np.stack((x, y, z), axis=3)
        particle_indices = kdtree.query_ball_point(field_pos, radius, workers=-1)

        f = np.zeros(f_shape)

        def metaball_eval(p1, p2s):
            if p2s is None:
                return 0.0
            r = np.clip(np.linalg.norm(p1 - p2s, axis=1) / radius, 0.0, 1.0)
            ans = (1.0 - r ** 3 * (r * (r * 6.0 - 15.0) + 10.0)).sum()
            print(p1, p2s, ans)
            exit(0)
            return ans

        for ii in range(resolution[0]):
            for jj in range(resolution[1]):
                for kk in range(resolution[2]):
                    idx = (ii, jj, kk)
                    p1 = np.array([x[idx], y[idx], z[idx]])
                    p2s = np.array([particle_pos[p2_idx] for p2_idx in particle_indices[idx]]) if particle_indices[
                        idx] else None
                    f[idx] = metaball_eval(p1, p2s)
        return f

    def reconstruct_mesh(self, model_output_dir, frame_index):
        obj_name = os.path.join(model_output_dir, 'models', f'{frame_index:06d}.obj')
        print(obj_name)
        f = self.metaball_scalar_field(self.reconstruct_resolution)
        print(f.shape)
        vertices, faces = mcubes.marching_cubes(f, self.reconstruct_threshold)
        print(vertices.shape, faces.shape)
        mcubes.export_obj(vertices, faces, obj_name)

    def frames2mp4(self, render_output_dir):
        # Convert frames to mp4 by cv2
        print(f'Converting frames to mp4, frame rate: {self.frame_rate}')
        frames = sorted(f for f in os.listdir(render_output_dir) if f.lower().endswith(".png"))
        video = cv2.VideoWriter(os.path.join(render_output_dir, 'video.mp4'),
                                cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                int(self.frame_rate + 0.5), self.resolution)
        for i, frame in enumerate(tqdm.tqdm(frames)):
            frame_image = cv2.imread(os.path.join(render_output_dir, frame), cv2.IMREAD_COLOR)
            video.write(frame_image)
        video.release()

    def render_1spp(self, frame_index, window, scene, camera, canvas, model_output_dir, render_output_dir):
        self.step()

        camera.position(2, 0.8, 1.6)
        camera.lookat(0.0, 0.4, 0)
        scene.set_camera(camera)

        scene.point_light(pos=(-1.2, 1.2, 2), color=(1, 1, 1))
        scene.ambient_light((0.5, 0.5, 0.5))

        scene.particles(centers=self.x, radius=self.point_radius, color=self.point_color)
        scene.lines(vertices=self.bounding_box_vertices, width=0.02, color=(1., 1., 0.))

        canvas.scene(scene)

        if self.save_frames and frame_index <= self.save_seconds * self.frame_rate:
            frame_name = os.path.join(render_output_dir, f'{frame_index:06d}.png')
            window.save_image(frame_name)
        if self.save_obj:
            self.reconstruct_mesh(model_output_dir, frame_index)

        window.show()

    def run(self, seconds=None):
        window = ti.ui.Window('MPM3D', self.resolution, vsync=True)
        canvas = window.get_canvas()
        canvas.set_background_color(self.background_color)
        scene = ti.ui.Scene()
        camera = ti.ui.Camera()

        self.initialize()

        output_dir = os.path.join(os.path.dirname(__file__), 'mpm3d-outputs')

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        render_output_dir = os.path.join(output_dir, 'renders')
        model_output_dir = os.path.join(output_dir, 'models')
        os.makedirs(render_output_dir, exist_ok=True)
        os.makedirs(model_output_dir, exist_ok=True)

        if seconds is not None:
            self.save_seconds = seconds
            for frame_index in tqdm.tqdm(range(int(self.save_seconds * self.frame_rate))):
                self.render_1spp(frame_index, window, scene, camera, canvas, model_output_dir, render_output_dir)
        else:
            frame_index = int(self.save_seconds * self.frame_rate)
            for frame_index in tqdm.tqdm(range(frame_index)):
                self.render_1spp(frame_index, window, scene, camera, canvas, model_output_dir, render_output_dir)
            while window.running:
                self.render_1spp(frame_index, window, scene, camera, canvas, model_output_dir, render_output_dir)
                frame_index += 1

        if self.save_frames:
            self.frames2mp4(render_output_dir)


if __name__ == '__main__':
    mpm3d_simulator = MPM3DSimulator()
    # mpm3d_simulator.run()
    mpm3d_simulator.frames2mp4('mpm3d-outputs/renders')
