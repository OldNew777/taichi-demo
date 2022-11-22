import taichi as ti


@ti.func
def clamp(x, min_val, max_val):
    return ti.max(min_val, ti.min(max_val, x))


def mpm3d():
    ti.init(arch=ti.cpu)

    # rendering parameters
    resolution = (512, 512)
    n_steps = 50
    n_grids = 64
    background_color = (0.2, 0.2, 0.4)
    point_color = (0.4, 0.6, 0.6)
    point_radius = 0.003

    # constant physical values for simulation
    n_particles = (n_grids ** 3) // 2
    dx = 1. / n_grids
    dt = 8e-5
    p_rho = 1.
    p_vol = (dx * 0.5) ** 3
    p_mass = p_vol * p_rho
    gravity = 9.8
    bound = 3
    E = 400.

    # values for record
    x = ti.Vector.field(n=3, dtype=float, shape=(n_particles,))
    v = ti.Vector.field(n=3, dtype=float, shape=(n_particles,))
    J = ti.field(dtype=float, shape=(n_particles,))
    C = ti.Matrix.field(n=3, m=3, dtype=float, shape=(n_particles,))
    grid_v = ti.Vector.field(n=3, dtype=float, shape=(n_grids, n_grids, n_grids))
    grid_m = ti.field(dtype=float, shape=(n_grids, n_grids, n_grids))

    @ti.func
    def get_grid_index(xyz):
        return clamp(ti.cast(xyz, ti.i32), 0, n_grids - 1)

    @ti.kernel
    def initialize():
        for index in ti.grouped(x):
            x[index] = [ti.random() * 0.4 + 0.2, ti.random() * 0.4 + 0.2, ti.random() * 0.4 + 0.2]
            v[index] = [0., 0., 0.]
            J[index] = 1.
            C[index] = ti.Matrix.zero(float, 3, 3)

    @ti.kernel
    def clear_grid():
        grid_v.fill(ti.Vector.zero(float, 3))
        grid_m.fill(0.)

    @ti.func
    def weight_kernel(x):
        Xp = x / dx
        base = ti.cast(Xp - 0.5, ti.i32)
        fx = Xp - base.cast(float)
        w_l = 0.5 * ti.pow((1.5 - fx), 2)
        w_c = 0.75 - ti.pow((fx - 1), 2)
        w_r = 0.5 * ti.pow((fx - 0.5), 2)
        w = ti.Matrix.rows([w_l, w_c, w_r])
        return base, fx, w

    @ti.kernel
    def point_to_grid():
        for index in ti.grouped(x):
            base, fx, w = weight_kernel(x[index])
            stress = -4. * dt * E * p_vol * (J[index] - 1.) / ti.pow(dx, 2)
            affine = ti.Matrix.diag(3, stress) + p_mass * C[index]
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    for k in ti.static(range(3)):
                        offset = ti.Vector([i, j, k], dt=int)
                        dpos = (offset.cast(float) - fx) * dx
                        weight = w[i, 0] * w[j, 1] * w[k, 2]
                        v_add = weight * (p_mass * v[index] + affine @ dpos)
                        gird_index_new = get_grid_index(base + offset)
                        ti.atomic_add(grid_v[gird_index_new], v_add)
                        ti.atomic_add(grid_m[gird_index_new], weight * p_mass)

    @ti.kernel
    def simulate_grid():
        for i, j, k in ti.ndrange(grid_v.shape[0], grid_v.shape[1], grid_v.shape[2]):
            m_temp = grid_m[i, j, k]
            v_temp = ti.select(m_temp > 0, grid_v[i, j, k] / m_temp, ti.Vector.zero(float, 3))
            # if m_temp <= 0:
            #     print(i, j, k, m_temp, '<= 0 =', m_temp > 0, '; v_temp =', v_temp)
            v_temp[1] -= dt * gravity
            index = ti.Vector([i, j, k], dt=int)
            # if (index < bound).any():
            #     print(index, '<', bound, '=', index < bound, '; v_temp =', v_temp)
            cond = (index < bound and v_temp < 0.) or (index > n_grids - bound and v_temp > 0.)
            v_temp = ti.select(cond, 0., v_temp)
            grid_v[i, j, k] = v_temp

    @ti.kernel
    def grid_to_point():
        for index in ti.grouped(x):
            base, fx, w = weight_kernel(x[index])
            new_v = ti.Vector.zero(float, 3)
            new_C = ti.Matrix.zero(float, 3, 3)
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    for k in ti.static(range(3)):
                        offset = ti.Vector([i, j, k], dt=int)
                        dpos = (offset.cast(float) - fx) * dx
                        weight = w[i, 0] * w[j, 1] * w[k, 2]
                        gird_index_new = get_grid_index(base + offset)
                        g_v = grid_v[gird_index_new]
                        new_v += weight * g_v
                        new_C += 4. * weight * g_v.outer_product(dpos) / ti.pow(dx, 2)
            v[index] = new_v
            x[index] += dt * new_v
            J[index] *= 1. + dt * new_C.trace()
            C[index] = new_C

            base, fx, w = weight_kernel(x[index])
            gird_index_new = get_grid_index(base)
            if (gird_index_new != base).any():
                print('base =', base, '; fx =', fx, '; ', gird_index_new, '!=', base)
                print('dt =', dt, '; new_v =', new_v, '; new_C =', new_C)

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

        camera.position(1.5, 0.8, 1.3)
        camera.lookat(0.0, 0.0, 0)
        scene.set_camera(camera)

        scene.point_light(pos=(-1.2, 1.2, 2), color=(1, 1, 1))
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.particles(centers=x, radius=point_radius, color=point_color)
        canvas.scene(scene)
        window.show()


if __name__ == '__main__':
    mpm3d()
