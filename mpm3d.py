import shutil
import os

import taichi as ti
import numpy as np

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2
import tqdm


@ti.func
def norm(x):
    return ti.sqrt(x.dot(x))


@ti.func
def in_sphere(pos, center, radius):
    distance = norm(pos - center)
    return distance < radius


@ti.func
def normal_sphere(pos, center):
    distance = norm(pos - center)
    normal = ti.select(distance > 1e-5, (pos - center) / distance, ti.Vector.zero(float, 3))
    return normal


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
        self.radius_insert = 0.03
        self.pos_insert = ti.Vector([0.2, 0.9, 0.7])
        self.v_insert = ti.Vector([1.5, -5.0, -1.2])

        # allocate larger space for particles to be inserted
        self.n_particles = (self.n_grids ** 3) // (2 ** 3)
        self.n_particles_init = self.n_particles
        self.n_particles_max = max(self.n_particles << 2, 1 << 20)
        print(f'Particles: {self.n_particles}/{self.n_particles_max}')

        # constant physical values for simulation
        self.dx = 1. / self.n_grids
        self.dt = 8e-5
        self.p_rho = 1.
        self.p_vol = (self.dx * 0.5) ** 3
        self.p_mass = self.p_vol * self.p_rho
        self.gravity = 9.8
        self.bound = 3
        self.E = 400.
        self.rebound_ratio = 0.25

        # values for record
        self.x = ti.Vector.field(n=3, dtype=float, shape=(self.n_particles_max,))
        self.v = ti.Vector.field(n=3, dtype=float, shape=(self.n_particles_max,))
        self.J = ti.field(dtype=float, shape=(self.n_particles_max,))
        self.C = ti.Matrix.field(n=3, m=3, dtype=float, shape=(self.n_particles_max,))
        self.status = ti.field(dtype=int, shape=(self.n_particles_max,))
        self.grid_v = ti.Vector.field(n=3, dtype=float, shape=(self.n_grids, self.n_grids, self.n_grids))
        self.grid_m = ti.field(dtype=float, shape=(self.n_grids, self.n_grids, self.n_grids))

        # rendering parameters
        self.save_frames = False
        self.save_seconds = 10.0
        self.fps = 1. / self.dt / self.n_steps
        self.resolution = (1024, 1024)
        self.background_color = (0.2, 0.2, 0.4)
        self.particles_color = [(0.4, 0.6, 0.6), (0.8, 0.2, 0.2)]
        self.particles_radius = 0.003
        self.reconstruct_threshold = 0.75
        self.reconstruct_resolution = (100, 100, 100)
        self.reconstruct_radius = 0.1

        # boundary
        self.bounding_box_vertices = ti.Vector.field(n=3, dtype=float, shape=(24,))
        self.n_spheres = 1
        self.sphere_centers = [ti.Vector.field(n=3, dtype=float, shape=(1,)) for _ in range(self.n_spheres)]
        self.sphere_radius = [0.15]
        self.sphere_colors = [(0.6, 0.6, 0.2)]
        self.int_cache = ti.field(dtype=int, shape=(1,))

    @ti.func
    def get_grid_index(self, xyz):
        return ti.math.clamp(ti.cast(xyz, ti.i32), 0, self.n_grids - 1)

    @ti.kernel
    def initialize(self):
        self.sphere_centers[0][0] = [0.3, 0.15, 0.6]

        self.v.fill([0., 0., 0.])
        self.J.fill(1.)
        self.C.fill(0.)
        for index in range(self.n_particles):
            self.status[index] = ParticleStatus.VALID
            self.int_cache[0] = 1
            while self.int_cache[0] > 0:
                # init particles in a smaller cube
                self.x[index] = [ti.random() * 0.4 + 0.2, ti.random() * 0.4 + 0.2, ti.random() * 0.4 + 0.2]
                self.int_cache[0] = -1
                # reject particles in spheres
                for i in ti.static(range(self.n_spheres)):
                    if in_sphere(self.x[index], self.sphere_centers[i][0], self.sphere_radius[i]):
                        self.int_cache[0] = 1

        for index in range(self.n_particles, self.n_particles_max):
            #  init particles to be added
            theta = ti.random() * 2 * np.pi
            x_offset = (ti.random() - 0.5) * self.v_insert[1] * self.dt * self.n_steps
            r = 10000000.
            while r > self.radius_insert:
                r = ti.abs(gaussian_random()) * self.radius_insert
            self.x[index] = [r * ti.cos(theta), 0., r * ti.sin(theta)] + self.pos_insert + x_offset
            self.v[index] = self.v_insert
            self.status[index] = ParticleStatus.INVALID

        # boundary box
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

    @ti.kernel
    def add_particles(self, offset: int, n: int):
        for index in range(offset, offset + n):
            # mark particles to be added as VALID
            self.status[index] = ParticleStatus.VALID

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
        # transfer particles to grids
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
            # gravity
            m_temp = self.grid_m[i, j, k]
            v_temp = ti.select(m_temp > 0, self.grid_v[i, j, k] / m_temp, ti.Vector.zero(float, 3))
            v_temp[1] -= self.dt * self.gravity

            # bump into boundary
            # axis aligned bounding box
            grid_index = ti.Vector([i, j, k], dt=int)
            cond = (grid_index < self.bound and v_temp < 0.) or (grid_index > self.n_grids - self.bound and v_temp > 0.)
            v_temp = ti.select(cond, -self.rebound_ratio * v_temp, v_temp)

            # bump into objects placed in the space (sphere)
            for index in ti.static(range(self.n_spheres)):
                sphere_center = self.sphere_centers[index][0]
                radius = self.sphere_radius[index]
                grid_x = (grid_index.cast(float) + 0.5) * self.dx
                normal = normal_sphere(grid_x, sphere_center)
                beta = v_temp.dot(normal)
                if in_sphere(grid_x, sphere_center, radius) and beta < 0.:
                    # v along normal direction will change
                    v_temp -= (1. + self.rebound_ratio) * beta * normal

            self.grid_v[i, j, k] = v_temp

    @ti.kernel
    def grid_to_point(self):
        # transfer grids to particles
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
        n = min(self.n_insert_per_step, self.n_particles_max - self.n_particles)
        self.add_particles(self.n_particles, n)
        self.n_particles += n
        for i in range(self.n_steps):
            self.substep()

    def frames2mp4(self, render_output_dir):
        # Convert frames to mp4 by cv2
        default_fps = 60
        frame_every = int(self.fps / default_fps + 0.5)
        print(f'Converting frames to mp4, FPS: {self.fps}')
        frames = sorted(f for f in os.listdir(render_output_dir) if f.lower().endswith(".png"))
        video = cv2.VideoWriter(os.path.join(render_output_dir, 'video.mp4'),
                                cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                default_fps, self.resolution)
        for i, frame in enumerate(tqdm.tqdm(frames)):
            if i % frame_every != 0:
                continue
            frame_image = cv2.imread(os.path.join(render_output_dir, frame), cv2.IMREAD_COLOR)
            assert frame_image.shape[:2] == self.resolution
            video.write(frame_image)
        video.release()

    def render_1spp(self, frame_index: int,
                    window: ti.ui.Window, scene: ti.ui.Scene, camera: ti.ui.Camera, canvas: ti.ui.Canvas,
                    model_output_dir, render_output_dir):
        self.step()

        camera.position(2, 0.8, 1.6)
        camera.lookat(0.0, 0.4, 0)
        scene.set_camera(camera)

        scene.point_light(pos=(-1.2, 1.2, 2), color=(1, 1, 1))
        scene.ambient_light((0.5, 0.5, 0.5))

        scene.particles(centers=self.x, radius=self.particles_radius, color=self.particles_color[0],
                        index_offset=0, index_count=self.n_particles_init)
        scene.particles(centers=self.x, radius=self.particles_radius, color=self.particles_color[1],
                        index_offset=self.n_particles_init, index_count=self.n_particles - self.n_particles_init)

        scene.lines(vertices=self.bounding_box_vertices, width=0.02, color=(1., 1., 0.))
        for i in range(self.n_spheres):
            scene.particles(centers=self.sphere_centers[i], radius=self.sphere_radius[i], color=self.sphere_colors[i])

        canvas.scene(scene)

        if self.save_frames and frame_index <= self.save_seconds * self.fps:
            frame_name = os.path.join(render_output_dir, f'{frame_index:06d}.png')
            window.save_image(frame_name)

    def run(self, seconds=None):
        print('Start initialization')
        self.initialize()
        print('Finish initialization')

        window = ti.ui.Window('MPM3D', self.resolution, vsync=True)
        canvas = window.get_canvas()
        canvas.set_background_color(self.background_color)
        scene = ti.ui.Scene()
        camera = ti.ui.Camera()

        output_dir = os.path.join(os.path.dirname(__file__), 'mpm3d-outputs')

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        render_output_dir = os.path.join(output_dir, 'renders')
        model_output_dir = os.path.join(output_dir, 'models')
        os.makedirs(render_output_dir, exist_ok=True)
        os.makedirs(model_output_dir, exist_ok=True)

        if seconds is not None:
            self.save_seconds = seconds
            frame_num = int(self.save_seconds * self.fps)
            for frame_index in tqdm.tqdm(range(frame_num)):
                self.render_1spp(frame_index, window, scene, camera, canvas, model_output_dir, render_output_dir)
                window.show()
        else:
            frame_index = 0
            while window.running:
                self.render_1spp(frame_index, window, scene, camera, canvas, model_output_dir, render_output_dir)
                window.show()
                frame_index += 1

        if window.running:
            window.destroy()

        if self.save_frames:
            self.frames2mp4(render_output_dir)


if __name__ == '__main__':
    mpm3d_simulator = MPM3DSimulator()
    # mpm3d_simulator.run(seconds=10.0)
    mpm3d_simulator.frames2mp4('D:/OldNew/LuisaRender/record/mpm3d/v2/renders')
