import taichi as ti


@ti.func
def clamp(x, min_val, max_val):
    return ti.max(min_val, ti.min(max_val, x))


@ti.data_oriented
class MPM3DSimulator:
    def __init__(self):
        ti.init(arch=ti.cpu)

        # rendering parameters
        self.resolution = (512, 512)
        self.n_steps = 50
        self.n_grids = 64
        self.background_color = (0.2, 0.2, 0.4)
        self.point_color = (0.4, 0.6, 0.6)
        self.point_radius = 0.003

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
        self.grid_v = ti.Vector.field(n=3, dtype=float, shape=(self.n_grids, self.n_grids, self.n_grids))
        self.grid_m = ti.field(dtype=float, shape=(self.n_grids, self.n_grids, self.n_grids))

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

    @ti.kernel
    def clear_grid(self):
        self.grid_v.fill([0., 0., 0.])
        self.grid_m.fill(0.)

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

    def run(self):
        window = ti.ui.Window('MPM3D', self.resolution, vsync=True)
        canvas = window.get_canvas()
        canvas.set_background_color(self.background_color)
        scene = ti.ui.Scene()
        camera = ti.ui.Camera()

        self.initialize()
        centers = ti.Vector.field(3, float, self.n_particles)

        while window.running:
            for i in range(self.n_steps):
                self.substep()
            centers.copy_from(self.x)

            camera.position(1.5, 0.8, 1.3)
            camera.lookat(0.0, 0.0, 0)
            scene.set_camera(camera)

            scene.point_light(pos=(-1.2, 1.2, 2), color=(1, 1, 1))
            scene.ambient_light((0.5, 0.5, 0.5))
            scene.particles(centers=centers, radius=self.point_radius, color=self.point_color)
            canvas.scene(scene)
            window.show()


if __name__ == '__main__':
    mpm3d_simulator = MPM3DSimulator()
    mpm3d_simulator.run()
