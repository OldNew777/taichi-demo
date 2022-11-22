import taichi as ti


def mpm3d():
    ti.init(arch=ti.cpu)

    # rendering parameters
    resolution = (512, 512)
    n_steps = 50
    len_grid = 64
    background_color = (0.2, 0.2, 0.4)
    point_color = (0.4, 0.6, 0.6)
    point_radius = 0.005

    # constant physical values for simulation
    n_particles = (len_grid ** 3) // 2
    dx = 1. / len_grid
    dt = 8e-5
    p_rho = 1.
    p_vol = (dx * 0.5) ** 3
    p_mass = p_vol * p_rho
    gravity = ti.Vector([0, -9.8, 0])
    bound = 3
    E = 400.

    # values for record
    n_grid = len_grid ** 3
    x = ti.Vector.field(n=3, dtype=float, shape=(n_particles,))
    v = ti.Vector.field(n=3, dtype=float, shape=(n_particles,))
    C = ti.Matrix.field(n=3, m=3, dtype=float, shape=(n_particles,))
    J = ti.field(dtype=float, shape=(n_particles,))
    grid_v = ti.Vector.field(n=3, dtype=float, shape=(n_grid,))
    grid_m = ti.field(dtype=float, shape=(n_grid,))

    @ti.kernel
    def initialize():
        for i in range(n_particles):
            x[i] = [ti.random() * 0.4 + 0.2, ti.random() * 0.4 + 0.2, ti.random() * 0.4 + 0.2]
            v[i] = [0., 0., 0.]
            J[i] = 1.
            C[i] = ti.Matrix.zero(float, 3, 3)

    @ti.kernel
    def clear_grid():
        grid_v.fill(ti.Vector.zero(float, 3))
        grid_m.fill(0.)

    @ti.kernel
    def point_to_grid():
        for index in ti.grouped(x):
            Xp = x[index] / dx
            base = ti.cast(Xp - 0.5, ti.i32)
            fx = Xp - base.cast(float)
            w = ti.Vector([0.5 * ((1.5 - fx) ** 2),
                           0.75 - ((fx - 1) ** 2),
                           0.5 * ((fx - 0.5) ** 2)])
            stress = -4. * dt * E * p_vol * (J[index] - 1.) / (dx ** 2)
            affine = ti.Matrix.one(float, 3, 3) * stress + p_mass * C[index]
            for i, j, k in ti.ndrange(3, 3, 3):
                offset = ti.Vector([i, j, k])
                dpos = (offset.cast(float) - fx) * dx
                weight = w[i][0] * w[j][1] * w[k][2]
                v_add = weight * (p_mass * v[index] + affine.dot(dpos))
                ti.atomic_add(grid_v[base + offset], v_add)
                ti.atomic_add(grid_m[base + offset], weight * p_mass)

    @ti.kernel
    def simulate_grid():
        for i in ti.grouped(grid_v):
            pass
            # if grid_m[i] > 0:
            #     grid_v[i] = grid_v[i] / grid_m[i] + dt * gravity
            # else:
            #     grid_v[i] = [0., 0., 0.]

    @ti.kernel
    def grid_to_point():
        pass

    def substep():
        clear_grid()
        point_to_grid()
        simulate_grid()
        grid_to_point()

    window = ti.ui.Window('MPM3D', resolution, vsync=True)
    canvas = window.get_canvas()
    canvas.set_background_color(background_color)
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()

    initialize()

    while window.running:
        for i in range(n_steps):
            substep()

        camera.position(1.5, 1.3, 1.2)
        camera.lookat(0.0, 0.0, 0)
        scene.set_camera(camera)

        scene.point_light(pos=(-1.2, 1.2, 2), color=(1, 1, 1))
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.particles(centers=x, radius=point_radius, color=point_color)
        canvas.scene(scene)
        window.show()


if __name__ == '__main__':
    mpm3d()
