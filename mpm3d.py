import taichi as ti
import numpy as np


@ti.func
def clamp(x, min_val, max_val):
    return ti.max(min_val, ti.min(max_val, x))


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

        # rendering parameters
        self.resolution = (512, 512)
        self.n_steps = 50
        self.n_grids = 64
        self.background_color = (0.2, 0.2, 0.4)
        self.point_color = (0.4, 0.6, 0.6)
        self.point_radius = 0.003

        self.n_insert_per_step = 3
        self.radius_insert_last = ti.field(dtype=float, shape=(1, ))
        self.radius_insert = 0.03
        self.radius_insert_delta = 0.05
        self.pos_insert = ti.Vector([0.2, 0.7, 0.7])
        self.v_init_insert = ti.Vector([0.25, -0.5, -0.2])

        # constant physical values for simulation
        self.n_particles = (self.n_grids ** 3) // 2
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

        self.insert_index = self.n_particles - 1

        self.grid_v = ti.Vector.field(n=3, dtype=float, shape=(self.n_grids, self.n_grids, self.n_grids))
        self.grid_m = ti.field(dtype=float, shape=(self.n_grids, self.n_grids, self.n_grids))

        # attribute cache
        self.insert_indexes = ti.field(dtype=int, shape=(self.n_insert_per_step,))

    @ti.func
    def get_grid_index(self, xyz):
        return clamp(ti.cast(xyz, ti.i32), 0, self.n_grids - 1)

    @ti.kernel
    def initialize(self):
        for index in ti.grouped(self.x):
            self.x[index] = [ti.random() * 0.4 + 0.2, ti.random() * 0.4 + 0.2, ti.random() * 0.4 + 0.2]
        self.v.fill([0., 0., 0.])
        self.J.fill(1.)
        self.C.fill(0.)
        self.status.fill(ParticleStatus.VALID)
        self.radius_insert_last[0] = 0.

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

    @ti.kernel
    def valid_n(self) -> int:
        n = ti.int32(0)
        for i in ti.grouped(self.status):
            if self.status[i] == ParticleStatus.VALID:
                ti.atomic_add(n, 1)
        return int(n)


    @ti.kernel
    def init_added_particles(self):
        for i in ti.grouped(self.insert_indexes):
            index = self.insert_indexes[i]
            theta = ti.random() * 2 * np.pi
            r = ti.abs(gaussian_random() * self.radius_insert)
            while ti.abs(r - self.radius_insert_last[0]) > self.radius_insert_delta:
                r = ti.abs(gaussian_random() * self.radius_insert)
            self.radius_insert_last[0] = r
            self.x[index] = [r * ti.cos(theta), 0., r * ti.sin(theta)] + self.pos_insert
            self.v[index] = self.v_init_insert
            self.J[index] = 1.
            self.C[index] = ti.Matrix.zero(float, 3, 3)
            self.status[index] = ParticleStatus.VALID

    def add_particles(self):
        n = self.n_insert_per_step
        if self.valid_n() + n > self.n_particles:
            new_size = self.n_particles + max(n, self.n_particles, 65536)
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
                for i in ti.static(range(3)):
                    for j in ti.static(range(3)):
                        for k in ti.static(range(3)):
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
                for i in ti.static(range(3)):
                    for j in ti.static(range(3)):
                        for k in ti.static(range(3)):
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

    def run(self):
        window = ti.ui.Window('MPM3D', self.resolution, vsync=True)
        canvas = window.get_canvas()
        canvas.set_background_color(self.background_color)
        scene = ti.ui.Scene()
        camera = ti.ui.Camera()

        self.initialize()

        while window.running:
            self.step()

            camera.position(1.5, 0.8, 1.3)
            camera.lookat(0.0, 0.0, 0)
            scene.set_camera(camera)

            scene.point_light(pos=(-1.2, 1.2, 2), color=(1, 1, 1))
            scene.ambient_light((0.5, 0.5, 0.5))
            scene.particles(centers=self.x, radius=self.point_radius, color=self.point_color)
            canvas.scene(scene)
            window.show()


if __name__ == '__main__':
    mpm3d_simulator = MPM3DSimulator()
    mpm3d_simulator.run()
