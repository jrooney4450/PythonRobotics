"""

Path tracking simulation with Stanley steering control and PID speed control.

author: Atsushi Sakai (@Atsushi_twi)

Ref:
    - [Stanley: The robot that won the DARPA grand challenge](http://isl.ecst.csuchico.edu/DOCS/darpa2005/DARPA%202005%20Stanley.pdf)
    - [Autonomous Automobile Path Tracking](https://www.ri.cmu.edu/pub_files/2009/2/Automatic_Steering_Methods_for_Autonomous_Automobile_Path_Tracking.pdf)

"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
sys.path.append("../../PathPlanning/CubicSpline/")

try:
    import cubic_spline_planner
except:
    raise

class State(object):
    """
    Class representing the state of a vehicle.

    :param x: (float) x-coordinate
    :param y: (float) y-coordinate
    :param yaw: (float) yaw angle
    :param v: (float) speed
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        """Instantiate the object."""
        super(State, self).__init__()
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

    def update(self, acceleration, delta):
        """
        Update the state of the vehicle.

        Stanley Control uses bicycle model.

        :param acceleration: (float) Acceleration
        :param delta: (float) Steering
        """
        delta = np.clip(delta, -max_steer, max_steer)

        self.x += self.v * np.cos(self.yaw) * dt
        self.y += self.v * np.sin(self.yaw) * dt
        self.yaw += self.v / L * np.tan(delta) * dt
        self.yaw = normalize_angle(self.yaw)
        self.v += acceleration * dt


def normalize(a):
    """
    Return the normal unit vector.

    param a: vector ([float])
    """
    normal = np.empty_like(a)
    normal[0] = -a[1]
    normal[1] = a[0]
    normal = normal / np.linalg.norm(normal)
    return normal

def pid_control(target, current):
    """
    Proportional control for the speed.

    :param target: (float)
    :param current: (float)
    :return: (float)
    """
    return Kp * (target - current)


def stanley_control(state, cx, cy, cyaw, idx_now, sta_before):
    """
    Stanley steering control.

    :param state: (State object)
    :param cx: ([float])
    :param cy: ([float])
    :param cyaw: ([float])
    :param idx_now: (int)
    :return: (float, int)
    """
    # idx_next, e_ct = calc_target_index(state, cx, cy)
    idx_next, e_ct, sta, sta_before = calc_next_index(state, cx, cy, idx_now, sta_before)

    # if idx_now >= idx_next:
    #     idx_next = idx_now

    # theta_e corrects the heading error
    theta_e = normalize_angle(cyaw[idx_next] - state.yaw)
    # theta_d corrects the cross track error
    theta_d = np.arctan2(k * e_ct, state.v)
    # Steering control
    delta = theta_e + theta_d

    return delta, idx_next, e_ct, sta, sta_before


def normalize_angle(angle):
    """
    Normalize an angle to [-pi, pi].

    :param angle: (float)
    :return: (float) Angle in radian in [-pi, pi]
    """
    while angle > np.pi:
        angle -= 2.0 * np.pi

    while angle < -np.pi:
        angle += 2.0 * np.pi

    return angle


def calc_target_index(state, cx, cy):
    """
    Compute index in the trajectory list of the target.

    :param state: (State object)
    :param cx: [float]
    :param cy: [float]
    :return: (int, float)
    """
    # Calc front axle position
    fx = state.x + L * np.cos(state.yaw)
    fy = state.y + L * np.sin(state.yaw)

    # Search nearest point index
    dx = [fx - icx for icx in cx]
    dy = [fy - icy for icy in cy]
    d = np.hypot(dx, dy)
    idx_now = np.argmin(d)

    # Project RMS error onto front axle vector
    front_axle_vec = [-np.cos(state.yaw + np.pi / 2),
                      -np.sin(state.yaw + np.pi / 2)]
    e_ct = np.dot([dx[idx_now], dy[idx_now]], front_axle_vec)

    return idx_now, e_ct

def calc_next_index(state, cx, cy, idx_now, sta_before):
    """
    Compute next index in the trajectory.

    :param state: (State object)
    :param cx: [float]
    :param cy: [float]
    :return: next_idx, e_ct (cross-track error), sta (station) [int, float, float]
    """
    # Calc front axle position
    fx = state.x + L * np.cos(state.yaw)
    fy = state.y + L * np.sin(state.yaw)

    idx_next = idx_now + 1
    r_T = np.array([cx[idx_now], cy[idx_now]]) # desired position vector
    r = np.array([fx, fy])

    sta_vect = np.array([cx[idx_next] - cx[idx_now], cy[idx_next] - cy[idx_now]]) # station vector, or vector from current point on path to next point on path
    sta = np.linalg.norm(sta_vect) # magnitude of the station vector
    a = sta_vect / np.linalg.norm(sta) # trajectory tangent unit vector

    # sta_before = sta
    # print('The station before is: {}'.format(sta_before))
    sta = np.dot((r - r_T), a)
    # print('The station is: {}'.format(sta))
    sta_inc = sta - sta_before # station is actually the projection of the position onto the tangent vector minus the previous amount
    if sta_inc < 0:
        sta_inc += ds
    # print('The station increment is: {}'.format(sta_inc))
    sta_before = sta

    normal = normalize(a)
    # print('The tangent vector is: {}'.format(a))
    # print('The normal vector is: {}'.format(normal))

    # Eq. 9 in the following paper:
    # http://ai.stanford.edu/~gabeh/papers/GNC08_QuadTraj.pdf
    e_ct = np.dot((r_T - r), normal) # cross-track position error

    b = np.array([fx - cx[idx_next], fy - cy[idx_next]]) # vector from next index point to current robot state
    # print('The next index to state vector is: {}'.format(b))

    angle = np.arccos(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    # print('The angle is: {}'.format(angle))
    if angle < (np.pi/2):
        # print("incremented the counter")
        idx_now += 1

    # print('idx_next: {}'.format(idx_next))
    # print('cross-track error: {}'.format(e_ct))

    return idx_now, e_ct, sta_inc, sta_before


def getLaneChange():
    ax = [0.0, 29.0, 30.0, 40.0, 41.0, 100.0]
    ay = [0.0,  0.0,  0.0,  3.0,  3.0,   3.0]
    return ax, ay, 'Lane Change Course'


def getFigureEight():
    r = 65.0
    ax = [0.0,     0.707*r, r,     0.707*r, 0.0,    -0.707*r, -r,    -0.707*r, 0.0,\
              0.707*r,  r,      0.707*r,  0.0,     -0.707*r, -r,     -0.707*r, 0.0]
    ay = [0.0, r - 0.707*r, r, r + 0.707*r, 2*r, r + 0.707*r,  r, r - 0.707*r, 0.0,\
         -r + 0.707*r, -r, -r - 0.707*r, -2*r, -r - 0.707*r, -r, -r + 0.707*r, 0.0]
    return ax, ay, 'Figure Eight Course'

def getRoadPath():
    ax = [0.0, 150.0,  100.0,   50.0,    0.0, -75.0, -125.0, -125.0, -100.0,   50.0,\
          200.0,  250.0, 300.0, 225.0, 200.0, 150.0,  50.0, -125.0, -100.0, 0.0]
    ay = [0.0, -50.0, -135.0, -110.0, -125.0, -35.0,  -35.0, -150.0, -190.0, -190.0,\
         -190.0, -150.0,  60.0, 130.0, 150.0,  60.0, 110.0,  120.0,    0.0, 0.0]
    return ax, ay, 'Road Course'

def main():
    """Plot an example of Stanley steering control on a cubic spline."""
    print("Starting main loop with k={}".format(k))
    state = State(x=0.0, y=0.5, yaw=0.0, v=0.0)

    # Chose path
    ax, ay, course_type = getLaneChange()
    # ax, ay, course_type = getFigureEight()
    # ax, ay, course_type = getRoadPath()

    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
        ax, ay, ds=ds)

    max_simulation_time = 200.0

    idx_last = len(cx) - 1
    time = 0.0
    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    v = [state.v]
    t = [0.0]
    e_ct_arr = []
    station = []
    sta_cum = 0.
    sta_before = 0.
    idx_now = 0

    while max_simulation_time >= time and idx_last > idx_now:
        ai = pid_control(target_speed, state.v)
        di, idx_now, e_ct, sta, sta_before = stanley_control(state, cx, cy, cyaw, idx_now, sta_before)
        state.update(ai, di)

        time += dt

        sta_cum += sta
        station.append(sta_cum)
        e_ct_arr.append(e_ct)
        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(time)

        # if show_animation:  # pragma: no cover
        #     plt.cla()
        #     plt.plot(cx, cy, ".r")
        #     # plt.plot(cx, cy, ".r", label="course")
        #     plt.plot(x, y, "-b")
        #     # plt.plot(x, y, "-b", label="trajectory")
        #     plt.plot(cx[idx_now], cy[idx_now], "xg", label="target")
        #     plt.axis("equal")
        #     plt.grid(True)
        #     plt.title("Speed[km/h]:" + str(state.v * 3.6)[:4])
        #     plt.pause(0.0001)

    # Test
    assert idx_last >= idx_now, "Cannot reach goal"

    if show_animation:  # pragma: no cover
        # Plot course and car path
        plt.figure(1)
        plt.plot(cx, cy, "-r")
        # plt.scatter(cx, cy, c="k", markersize=0.1)
        # plt.plot(x, y, "-b")
        plt.legend()
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.title("{}".format(course_type))
        plt.axis("equal")
        plt.grid(True)

        # Plot errors
        plt.figure(0)
        plt.plot(station, e_ct_arr, c=color, label="k={}".format(k))
        plt.xlabel("Station[m]")
        plt.ylabel("Cross-Track Error[m]")
        plt.title("Stanley at {}m/s on {}".format(target_speed, course_type))
        plt.grid(True)
        # plt.legend(loc="best")

    return 0

if __name__ == '__main__':
    Kp = 1.0  # speed proportional gain
    factor = 15
    dt = 0.1 / factor  # [s] time difference
    L = 2.9  # [m] Wheel base of vehicle
    max_steer = np.radians(30.0)  # [rad] max steering angle
    show_animation = True
    ds = 5 # for the cublic spline planner

    target_speed = 20.0 # [m/s]
    
    k = 1.0  # control gain
    color = 'black'
    main()

    k = 2.0
    color = 'blue'
    main()

    k = 4.0
    color = 'cyan'
    main()

    k = 6.0
    color = 'limegreen'
    main()

    k = 8.0
    color = 'pink'
    main()

    k = 10.0
    color = 'red'
    main()

    plt.legend(loc="best")
    plt.savefig('images/stanley_lane_change_{}.png'.format(target_speed))
    # plt.savefig('images/stanley_figure_eight_{}.png'.format(target_speed))
    # plt.savefig('images/stanley_road_{}.png'.format(target_speed))
    plt.show()