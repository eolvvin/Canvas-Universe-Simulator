import taichi as ti
import math

ti.init(arch=ti.gpu)

# ============================================================
# CONFIGURATION
# ============================================================
GRID_SIZE = 64
DT = 0.03
C_EFF = 0.00446
D_EFF = 0.00284
PHI_0 = 1.0
T_ST = 4.0
SIGMA = 0.5
AMPLIFICATION = 1.015
MAX_AMP = 80.0

N_FIELDS = 8

osc_pos = ti.Vector.field(3, dtype=ti.f32, shape=(N_FIELDS, 2))

phi = ti.field(dtype=ti.f32, shape=(N_FIELDS, GRID_SIZE, GRID_SIZE, GRID_SIZE))
phi_old = ti.field(dtype=ti.f32, shape=(N_FIELDS, GRID_SIZE, GRID_SIZE, GRID_SIZE))
phi_new = ti.field(dtype=ti.f32, shape=(N_FIELDS, GRID_SIZE, GRID_SIZE, GRID_SIZE))

voxel = ti.field(dtype=ti.i32, shape=(GRID_SIZE, GRID_SIZE, GRID_SIZE))
voxel_age = ti.field(dtype=ti.f32, shape=(GRID_SIZE, GRID_SIZE, GRID_SIZE))

particle_pos = ti.Vector.field(3, dtype=ti.f32, shape=(1000))
particle_vel = ti.Vector.field(3, dtype=ti.f32, shape=(1000))
particle_active = ti.field(dtype=ti.i32, shape=(1000))
particle_count = ti.field(dtype=ti.i32, shape=())

pixels = ti.field(dtype=ti.f32, shape=(1024, 1024, 3))

view_zoom = ti.field(dtype=ti.f32, shape=())
view_offset_x = ti.field(dtype=ti.f32, shape=())
view_offset_y = ti.field(dtype=ti.f32, shape=())

# ============================================================
# HELPER FUNCTIONS
# ============================================================

@ti.func
def sgn(x):
    result = 0.0
    if x > 0.0:
        result = 1.0
    elif x < 0.0:
        result = -1.0
    return result

# ============================================================
# INITIALIZATION
# ============================================================

@ti.kernel
def initialize():
    center = GRID_SIZE // 2
    c = float(center)
    q = float(GRID_SIZE // 4)
    
    osc_pos[0, 0] = ti.Vector([c, c, c - q])
    osc_pos[0, 1] = ti.Vector([c, c, c + q])
    osc_pos[1, 0] = ti.Vector([c, c, c - q])
    osc_pos[1, 1] = ti.Vector([c, c, c + q])
    
    osc_pos[2, 0] = ti.Vector([c - q, c, c])
    osc_pos[2, 1] = ti.Vector([c + q, c, c])
    osc_pos[3, 0] = ti.Vector([c - q, c, c])
    osc_pos[3, 1] = ti.Vector([c + q, c, c])
    
    osc_pos[4, 0] = ti.Vector([c, c - q, c])
    osc_pos[4, 1] = ti.Vector([c, c + q, c])
    osc_pos[5, 0] = ti.Vector([c, c - q, c])
    osc_pos[5, 1] = ti.Vector([c, c + q, c])
    
    osc_pos[6, 0] = ti.Vector([c, c, c - q])
    osc_pos[6, 1] = ti.Vector([c, c, c + q])
    osc_pos[7, 0] = ti.Vector([c, c, c - q])
    osc_pos[7, 1] = ti.Vector([c, c, c + q])
    
    for f, i, j, k in phi:
        phi[f, i, j, k] = 0.0
        phi_old[f, i, j, k] = 0.0
        phi_new[f, i, j, k] = 0.0
    
    for i, j, k in voxel:
        voxel[i, j, k] = 0
        voxel_age[i, j, k] = 0.0
    
    for p in range(1000):
        particle_active[p] = 0
    
    particle_count[None] = 0
    
    view_zoom[None] = 1.5
    view_offset_x[None] = 0.0
    view_offset_y[None] = 0.0

# ============================================================
# OSCILLATOR WAVE PROPAGATION
# ============================================================

@ti.kernel
def step_oscillators(t: ti.f32, amp: ti.f32):
    freq = 2.0 * 3.14159265 * 0.1
    phase_shift = 3.14159265 * 0.5
    for f in range(N_FIELDS):
        phase0 = ti.sin(t * freq)
        phase1 = ti.sin(t * freq + phase_shift)
        
        pos0 = osc_pos[f, 0]
        pos1 = osc_pos[f, 1]
        
        for i, j, k in ti.ndrange(GRID_SIZE, GRID_SIZE, GRID_SIZE):
            fi = float(i)
            fj = float(j)
            fk = float(k)
            d0 = ti.sqrt((fi - pos0[0])**2 + (fj - pos0[1])**2 + (fk - pos0[2])**2)
            d1 = ti.sqrt((fi - pos1[0])**2 + (fj - pos1[1])**2 + (fk - pos1[2])**2)
            
            if d0 > 0.1 and d1 > 0.1:
                wave0 = phase0 * ti.exp(-d0 * 0.015) / (d0 * 0.08 + 0.5)
                wave1 = phase1 * ti.exp(-d1 * 0.015) / (d1 * 0.08 + 0.5)
                phi[f, i, j, k] = phi[f, i, j, k] * 0.7 + (wave0 + wave1) * PHI_0 * amp * 0.3

@ti.kernel
def step_uwe():
    scale = float(GRID_SIZE * GRID_SIZE) / 100.0
    for f, i, j, k in phi:
        if i > 1 and i < GRID_SIZE-2 and j > 1 and j < GRID_SIZE-2 and k > 1 and k < GRID_SIZE-2:
            lap = (
                phi[f, i+1, j, k] + phi[f, i-1, j, k] +
                phi[f, i, j+1, k] + phi[f, i, j-1, k] +
                phi[f, i, j, k+1] + phi[f, i, j, k-1] -
                6.0 * phi[f, i, j, k]
            ) / scale
            
            nonlinear = (phi[f, i, j, k] - D_EFF * PHI_0 * sgn(phi[f, i, j, k])) / C_EFF
            phi_new[f, i, j, k] = 2.0 * phi[f, i, j, k] - phi_old[f, i, j, k] + DT * DT * (lap + nonlinear)

@ti.kernel
def swap_fields():
    for f, i, j, k in phi:
        phi_old[f, i, j, k] = phi[f, i, j, k]
        phi[f, i, j, k] = phi_new[f, i, j, k]

@ti.kernel
def detect_threshold_and_particles():
    for i, j, k in voxel:
        space_amp = 0.0
        for f in range(2, 8):
            space_amp += ti.abs(phi[f, i, j, k])
        space_amp /= 6.0
        
        time_amp = (ti.abs(phi[0, i, j, k]) + ti.abs(phi[1, i, j, k])) / 2.0
        
        intensity = space_amp * time_amp / (SIGMA * SIGMA)
        if intensity > T_ST:
            if voxel[i, j, k] == 0:
                voxel[i, j, k] = 1
                voxel_age[i, j, k] = 0.0
                
                pid = ti.atomic_add(particle_count[None], 1)
                if pid < 1000:
                    particle_pos[pid] = ti.Vector([float(i), float(j), float(k)])
                    vel_x = ti.sin(float(i * j * 137 + pid * 73)) * 0.5
                    vel_y = ti.sin(float(j * k * 271 + pid * 193)) * 0.5
                    vel_z = ti.sin(float(k * i * 419 + pid * 311)) * 0.5
                    particle_vel[pid] = ti.Vector([vel_x, vel_y, vel_z])
                    particle_active[pid] = 1
            else:
                voxel_age[i, j, k] = voxel_age[i, j, k] + 0.016

@ti.kernel
def update_particles():
    center = float(GRID_SIZE // 2)
    for p in range(1000):
        if particle_active[p] == 1:
            pos = particle_pos[p]
            to_center = ti.Vector([center, center, center]) - pos
            dist = to_center.norm() + 0.1
            gravity = to_center.normalized() * 0.01 / (dist * 0.1)
            
            new_vel = particle_vel[p] + gravity * 0.1
            new_pos = pos + new_vel * 0.5
            
            for d in ti.static(range(3)):
                if new_pos[d] < 0.0:
                    new_pos[d] = 0.0
                    new_vel[d] = ti.abs(new_vel[d]) * 0.8
                if new_pos[d] > float(GRID_SIZE - 1):
                    new_pos[d] = float(GRID_SIZE - 1)
                    new_vel[d] = -ti.abs(new_vel[d]) * 0.8
            
            particle_pos[p] = new_pos
            particle_vel[p] = new_vel
            
            if dist < 2.0:
                particle_active[p] = 0

@ti.kernel
def count_voxels() -> ti.i32:
    count = 0
    for i, j, k in voxel:
        if voxel[i, j, k] == 1:
            count += 1
    return count

@ti.kernel
def count_active_particles() -> ti.i32:
    count = 0
    for p in range(1000):
        if particle_active[p] == 1:
            count += 1
    return count

# ============================================================
# VISUALIZATION
# ============================================================

@ti.kernel
def render_slice(slice_axis: ti.i32, slice_pos: ti.i32, show_field: ti.i32):
    zoom = view_zoom[None]
    off_x = view_offset_x[None]
    off_y = view_offset_y[None]
    
    for px, py in ti.ndrange(1024, 1024):
        gx = (float(px) / 1024.0 - 0.5) / zoom + off_x + 0.5
        gy = (float(py) / 1024.0 - 0.5) / zoom + off_y + 0.5
        
        gi = slice_pos
        gj = ti.cast(gx * GRID_SIZE, ti.i32)
        gk = ti.cast(gy * GRID_SIZE, ti.i32)
        
        if slice_axis == 1:
            gi = ti.cast(gx * GRID_SIZE, ti.i32)
            gj = slice_pos
            gk = ti.cast(gy * GRID_SIZE, ti.i32)
        
        if slice_axis == 2:
            gi = ti.cast(gx * GRID_SIZE, ti.i32)
            gj = ti.cast(gy * GRID_SIZE, ti.i32)
            gk = slice_pos
        
        r = 0.02
        g = 0.02
        b = 0.08
        
        if 0 <= gi < GRID_SIZE and 0 <= gj < GRID_SIZE and 0 <= gk < GRID_SIZE:
            space_amp = 0.0
            for f in range(2, 8):
                space_amp += ti.abs(phi[f, gi, gj, gk])
            space_amp /= 6.0
            time_amp = (ti.abs(phi[0, gi, gj, gk]) + ti.abs(phi[1, gi, gj, gk])) / 2.0
            intensity = space_amp * time_amp * 0.3
            b = b + intensity * 0.6
            
            v = voxel[gi, gj, gk]
            if v == 1:
                age = voxel_age[gi, gj, gk]
                brightness = 0.3 + age * 0.05
                r = r + brightness
                g = g + brightness * 0.7
                b = b + brightness * 0.3
        
        for p in range(1000):
            if particle_active[p] == 1:
                ppos = particle_pos[p]
                pgi = ti.cast(ppos[0], ti.i32)
                pgj = ti.cast(ppos[1], ti.i32)
                pgk = ti.cast(ppos[2], ti.i32)
                
                on_slice = False
                if slice_axis == 0 and pgi == slice_pos:
                    on_slice = True
                if slice_axis == 1 and pgj == slice_pos:
                    on_slice = True
                if slice_axis == 2 and pgk == slice_pos:
                    on_slice = True
                
                if on_slice:
                    ppx = 0.0
                    ppy = 0.0
                    if slice_axis == 0:
                        ppx = ppos[1] / GRID_SIZE
                        ppy = ppos[2] / GRID_SIZE
                    elif slice_axis == 1:
                        ppx = ppos[0] / GRID_SIZE
                        ppy = ppos[2] / GRID_SIZE
                    else:
                        ppx = ppos[0] / GRID_SIZE
                        ppy = ppos[1] / GRID_SIZE
                    
                    dx = gx - ppx
                    dy = gy - ppy
                    dist = ti.sqrt(dx*dx + dy*dy)
                    if dist < 0.015:
                        r = 1.0
                        g = 1.0
                        b = 0.3
        
        pixels[px, py, 0] = r if r < 1.0 else 1.0
        pixels[px, py, 1] = g if g < 1.0 else 1.0
        pixels[px, py, 2] = b if b < 1.0 else 1.0

# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("CANVAS MODEL — Universe Simulator with Particles")
    print("=" * 60)
    print("CONTROLS:")
    print("  Q/E        = Zoom In / Out")
    print("  Arrows     = Pan")
    print("  A/S/D      = YZ / XZ / XY slice")
    print("  W/X        = Move slice forward/back")
    print("  1/2        = Combined / Time field view")
    print("  R          = Reset")
    print("  Space      = Pause")
    print("=" * 60)
    print("WHAT YOU'LL SEE:")
    print("  Blue glow  = Wave fields")
    print("  Warm glow  = Spacetime voxels (older = brighter)")
    print("  Yellow dots = Particles moving under gravity")
    print("=" * 60)
    
    initialize()
    
    window = ti.ui.Window("Canvas Universe - Q/E zoom, see particles!", (1024, 1024), vsync=True)
    canvas = window.get_canvas()
    
    t = 0.0
    frame = 0
    slice_pos = GRID_SIZE // 2
    view_axis = 2
    show_field = -1
    amp = 1.0
    paused = False
    
    while window.running:
        if window.is_pressed('q'):
            view_zoom[None] = view_zoom[None] * 1.05
        if window.is_pressed('e'):
            view_zoom[None] = view_zoom[None] * 0.95
        
        if window.is_pressed(ti.ui.LEFT):
            view_offset_x[None] = view_offset_x[None] - 0.03 / view_zoom[None]
        if window.is_pressed(ti.ui.RIGHT):
            view_offset_x[None] = view_offset_x[None] + 0.03 / view_zoom[None]
        if window.is_pressed(ti.ui.UP):
            view_offset_y[None] = view_offset_y[None] - 0.03 / view_zoom[None]
        if window.is_pressed(ti.ui.DOWN):
            view_offset_y[None] = view_offset_y[None] + 0.03 / view_zoom[None]
        
        view_zoom[None] = max(0.3, min(20.0, view_zoom[None]))
        
        if window.is_pressed('a'):
            view_axis = 0; slice_pos = GRID_SIZE // 2
            view_offset_x[None] = 0.0; view_offset_y[None] = 0.0
        if window.is_pressed('s'):
            view_axis = 1; slice_pos = GRID_SIZE // 2
            view_offset_x[None] = 0.0; view_offset_y[None] = 0.0
        if window.is_pressed('d'):
            view_axis = 2; slice_pos = GRID_SIZE // 2
            view_offset_x[None] = 0.0; view_offset_y[None] = 0.0
        
        if window.is_pressed('w'):
            slice_pos = min(slice_pos + 1, GRID_SIZE - 1)
        if window.is_pressed('x'):
            slice_pos = max(slice_pos - 1, 0)
        
        if window.is_pressed('1'):
            show_field = -1
        if window.is_pressed('2'):
            show_field = 0
        
        if window.is_pressed('r'):
            initialize()
            amp = 1.0; frame = 0; t = 0.0; paused = False
        
        if window.is_pressed(ti.ui.SPACE):
            paused = not paused
            ti.sleep(200)
        
        if not paused:
            amp = min(amp * AMPLIFICATION, MAX_AMP)
        
        step_oscillators(t, amp)
        if frame % 2 == 0:
            step_uwe()
            swap_fields()
        if frame % 5 == 0:
            detect_threshold_and_particles()
        if frame % 3 == 0:
            update_particles()
        
        render_slice(view_axis, slice_pos, show_field)
        canvas.set_image(pixels)
        window.show()
        
        t += DT
        frame += 1
        
        if frame % 60 == 0:
            vc = count_voxels()
            pc = count_active_particles()
            status = "PAUSED" if paused else "RUN"
            print(f"F{frame} [{status}] amp={amp:.0f} zoom={view_zoom[None]:.1f} voxels={vc} particles={pc}")

if __name__ == "__main__":
    main()