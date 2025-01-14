"""

Path tracking simulation with iterative linear model predictive control for speed and steer control

author: Atsushi Sakai (@Atsushi_twi)

"""
import matplotlib.pyplot as plt
import cvxpy
import math
import numpy as np
import sys
sys.path.append("../../PathPlanning/CubicSpline/")

try:
    import cubic_spline_planner
except:
    raise

class State:
    """
    vehicle state class
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.predelta = None


def pi_2_pi(angle):
    while(angle > math.pi):
        angle = angle - 2.0 * math.pi

    while(angle < -math.pi):
        angle = angle + 2.0 * math.pi

    return angle


def get_linear_model_matrix(v, phi, delta):

    A = np.zeros((NX, NX))
    A[0, 0] = 1.0
    A[1, 1] = 1.0
    A[2, 2] = 1.0
    A[3, 3] = 1.0
    A[0, 2] = DT * math.cos(phi)
    A[0, 3] = - DT * v * math.sin(phi)
    A[1, 2] = DT * math.sin(phi)
    A[1, 3] = DT * v * math.cos(phi)
    A[3, 2] = DT * math.tan(delta) / WB

    B = np.zeros((NX, NU))
    B[2, 0] = DT
    B[3, 1] = DT * v / (WB * math.cos(delta) ** 2)

    C = np.zeros(NX)
    C[0] = DT * v * math.sin(phi) * phi
    C[1] = - DT * v * math.cos(phi) * phi
    C[3] = - DT * v * delta / (WB * math.cos(delta) ** 2)

    return A, B, C


def plot_car(x, y, yaw, steer=0.0, cabcolor="-r", truckcolor="-k"):  # pragma: no cover

    outline = np.array([[-BACKTOWHEEL, (LENGTH - BACKTOWHEEL), (LENGTH - BACKTOWHEEL), -BACKTOWHEEL, -BACKTOWHEEL],
                        [WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

    fr_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
                         [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD]])

    rr_wheel = np.copy(fr_wheel)

    fl_wheel = np.copy(fr_wheel)
    fl_wheel[1, :] *= -1
    rl_wheel = np.copy(rr_wheel)
    rl_wheel[1, :] *= -1

    Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                     [-math.sin(yaw), math.cos(yaw)]])
    Rot2 = np.array([[math.cos(steer), math.sin(steer)],
                     [-math.sin(steer), math.cos(steer)]])

    fr_wheel = (fr_wheel.T.dot(Rot2)).T
    fl_wheel = (fl_wheel.T.dot(Rot2)).T
    fr_wheel[0, :] += WB
    fl_wheel[0, :] += WB

    fr_wheel = (fr_wheel.T.dot(Rot1)).T
    fl_wheel = (fl_wheel.T.dot(Rot1)).T

    outline = (outline.T.dot(Rot1)).T
    rr_wheel = (rr_wheel.T.dot(Rot1)).T
    rl_wheel = (rl_wheel.T.dot(Rot1)).T

    outline[0, :] += x
    outline[1, :] += y
    fr_wheel[0, :] += x
    fr_wheel[1, :] += y
    rr_wheel[0, :] += x
    rr_wheel[1, :] += y
    fl_wheel[0, :] += x
    fl_wheel[1, :] += y
    rl_wheel[0, :] += x
    rl_wheel[1, :] += y

    plt.plot(np.array(outline[0, :]).flatten(),
             np.array(outline[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fr_wheel[0, :]).flatten(),
             np.array(fr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rr_wheel[0, :]).flatten(),
             np.array(rr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fl_wheel[0, :]).flatten(),
             np.array(fl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rl_wheel[0, :]).flatten(),
             np.array(rl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(x, y, "*")


def update_state(state, a, delta):

    # input check
    if delta >= MAX_STEER:
        delta = MAX_STEER
    elif delta <= -MAX_STEER:
        delta = -MAX_STEER

    state.x = state.x + state.v * math.cos(state.yaw) * DT
    state.y = state.y + state.v * math.sin(state.yaw) * DT
    state.yaw = state.yaw + state.v / WB * math.tan(delta) * DT
    state.v = state.v + a * DT

    if state. v > MAX_SPEED:
        state.v = MAX_SPEED
    elif state. v < MIN_SPEED:
        state.v = MIN_SPEED

    return state


def get_nparray_from_matrix(x):
    return np.array(x).flatten()


def calc_nearest_index(state, cx, cy, cyaw, pind):

    dx = [state.x - icx for icx in cx[pind:(pind + N_IND_SEARCH)]]
    dy = [state.y - icy for icy in cy[pind:(pind + N_IND_SEARCH)]]

    d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

    mind = min(d)
    # print('mind is: {}'.format(mind))

    ind = d.index(mind) + pind

    mind = math.sqrt(mind)

    dxl = cx[ind] - state.x
    dyl = cy[ind] - state.y

    angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
    if angle < 0:
        mind *= -1

    return ind, mind

def normalize(a):
    """
    Return the n unit vector.

    param a: vector ([float])
    """
    n = np.empty_like(a)
    n[0] = -a[1]
    n[1] = a[0]
    n = n / np.linalg.norm(n)
    return n

def calc_error(state, cx, cy, idx_now, sta_before):
    """
    Compute next index in the trajectory. Main functional changes are here!

    :param state: (State object)
    :param cx: [float]
    :param cy: [float]
    :param idx_now: [int]
    :param sta_before: [float]
    :return: e_ct (cross-track error), sta_inc (station increment), 
        sta_before (station before) [float, float, float]
    """
    # TODO: idx_next goes out of possible array at end of course due to
    # car's ability to go in reverse

    # Calculate front axle position and use for index incrementing
    fx = state.x + LENGTH * np.cos(state.yaw)
    fy = state.y + LENGTH * np.sin(state.yaw)

    idx_next = idx_now + 1
    r_T = np.array([cx[idx_now], cy[idx_now]]) # desired position vector
    r = np.array([fx, fy])

    # Course vector, or vector along current segment of the course
    course_vect = np.array([cx[idx_next] - cx[idx_now], cy[idx_next] - cy[idx_now]])
    sta = np.linalg.norm(course_vect) # Magnitude of the course vector
    t = course_vect / np.linalg.norm(sta) # Tangent unit vector

    # Calculate station - the projection of the vehicle position along the course segment
    sta = np.dot((r - r_T), t)

    # Calculate station increment - the amount of station increased per function call
    sta_inc = sta - sta_before
    if sta_inc < 0:
        sta_inc += DL
    
    # Store the amount of station for use upon consequtive function call
    sta_before = sta

    # Find the n unit vector
    n = normalize(t)

    # Find cross-track position error - Eq. 9 in the following paper:
    # http://ai.stanford.edu/~gabeh/papers/GNC08_QuadTraj.pdf
    e_ct = np.dot((r - r_T), n)

    # Find vector from next index point to state of robot front axle
    b = np.array([fx - cx[idx_next], fy - cy[idx_next]]) 

    # Find angle between robot state
    angle = np.arccos(np.dot(t, b) / (np.linalg.norm(t) * np.linalg.norm(b)))

    # The waypoint will increment once the vehicle passes the n vector
    if angle < (np.pi/2):
        idx_now += 1

    return e_ct, sta_inc, sta_before

def predict_motion(x0, oa, od, xref):
    xbar = xref * 0.0
    for i, _ in enumerate(x0):
        xbar[i, 0] = x0[i]

    state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
    for (ai, di, i) in zip(oa, od, range(1, T + 1)):
        state = update_state(state, ai, di)
        xbar[0, i] = state.x
        xbar[1, i] = state.y
        xbar[2, i] = state.v
        xbar[3, i] = state.yaw

    return xbar


def iterative_linear_mpc_control(xref, x0, dref, oa, od):
    """
    MPC contorl with updating operational point iteraitvely
    """

    if oa is None or od is None:
        oa = [0.0] * T
        od = [0.0] * T

    for i in range(MAX_ITER):
        xbar = predict_motion(x0, oa, od, xref)
        poa, pod = oa[:], od[:]
        oa, od, ox, oy, oyaw, ov = linear_mpc_control(xref, xbar, x0, dref)
        du = sum(abs(oa - poa)) + sum(abs(od - pod))  # calc u change value
        if du <= DU_TH:
            break
    else:
        print("Iterative is max iter")

    return oa, od, ox, oy, oyaw, ov


def linear_mpc_control(xref, xbar, x0, dref):
    """
    linear mpc control

    xref: reference point
    xbar: operational point
    x0: initial state
    dref: reference steer angle
    """

    x = cvxpy.Variable((NX, T + 1))
    u = cvxpy.Variable((NU, T))

    cost = 0.0
    constraints = []

    for t in range(T):
        cost += cvxpy.quad_form(u[:, t], R)

        if t != 0:
            cost += cvxpy.quad_form(xref[:, t] - x[:, t], Q)

        A, B, C = get_linear_model_matrix(
            xbar[2, t], xbar[3, t], dref[0, t])
        constraints += [x[:, t + 1] == A * x[:, t] + B * u[:, t] + C]

        if t < (T - 1):
            cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], Rd)
            constraints += [cvxpy.abs(u[1, t + 1] - u[1, t]) <=
                            MAX_DSTEER * DT]

    cost += cvxpy.quad_form(xref[:, T] - x[:, T], Qf)

    constraints += [x[:, 0] == x0]
    constraints += [x[2, :] <= MAX_SPEED]
    constraints += [x[2, :] >= MIN_SPEED]
    constraints += [cvxpy.abs(u[0, :]) <= MAX_ACCEL]
    constraints += [cvxpy.abs(u[1, :]) <= MAX_STEER]

    prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    prob.solve(solver=cvxpy.ECOS, verbose=False)

    if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
        ox = get_nparray_from_matrix(x.value[0, :])
        oy = get_nparray_from_matrix(x.value[1, :])
        ov = get_nparray_from_matrix(x.value[2, :])
        oyaw = get_nparray_from_matrix(x.value[3, :])
        oa = get_nparray_from_matrix(u.value[0, :])
        odelta = get_nparray_from_matrix(u.value[1, :])

    else:
        print("Error: Cannot solve mpc..")z
        oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None

    return oa, odelta, ox, oy, oyaw, ov


def calc_ref_trajectory(state, cx, cy, cyaw, ck, sp, pind):
    xref = np.zeros((NX, T + 1))
    dref = np.zeros((1, T + 1))
    ncourse = len(cx)

    ind, _ = calc_nearest_index(state, cx, cy, cyaw, pind)

    if pind >= ind:
        ind = pind

    xref[0, 0] = cx[ind]
    xref[1, 0] = cy[ind]
    xref[2, 0] = sp[ind]
    xref[3, 0] = cyaw[ind]
    dref[0, 0] = 0.0  # steer operational point should be 0

    travel = 0.0

    for i in range(T + 1):
        travel += abs(state.v) * DT
        dind = int(round(travel / DL))

        if (ind + dind) < ncourse:
            xref[0, i] = cx[ind + dind]
            xref[1, i] = cy[ind + dind]
            xref[2, i] = sp[ind + dind]
            xref[3, i] = cyaw[ind + dind]
            dref[0, i] = 0.0
        else:
            xref[0, i] = cx[ncourse - 1]
            xref[1, i] = cy[ncourse - 1]
            xref[2, i] = sp[ncourse - 1]
            xref[3, i] = cyaw[ncourse - 1]
            dref[0, i] = 0.0

    return xref, ind, dref


def check_goal(state, goal, tind, nind):

    # check goal
    dx = state.x - goal[0]
    dy = state.y - goal[1]
    d = math.hypot(dx, dy)

    isgoal = (d <= GOAL_DIS)

    if abs(tind - nind) >= 5:
        isgoal = False

    isstop = (abs(state.v) <= STOP_SPEED)

    # if isgoal and isstop:
    #     return True

    if isgoal:
        return True

    return False


def do_simulation(cx, cy, cyaw, ck, sp, initial_state, course_name):
    """
    Simulation

    cx: course x position list
    cy: course y position list
    cy: course yaw position list
    ck: course curvature list
    sp: speed profile

    """

    goal = [cx[-1], cy[-1]]

    state = initial_state

    # initial yaw compensation
    if state.yaw - cyaw[0] >= math.pi:
        state.yaw -= math.pi * 2.0
    elif state.yaw - cyaw[0] <= -math.pi:
        state.yaw += math.pi * 2.0

    time = 0.0
    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    v = [state.v]
    t = [0.0]
    d = [0.0]
    a = [0.0]
    target_ind, _ = calc_nearest_index(state, cx, cy, cyaw, 0)

    odelta, oa = None, None

    cyaw = smooth_yaw(cyaw)

    sta_before = 0. 
    e_ct_arr = []
    station = []
    sta_cum = 0.
    sta_before = 0. # assume car starts at 0

    while MAX_TIME >= time:
        xref, target_ind, dref = calc_ref_trajectory(
            state, cx, cy, cyaw, ck, sp, target_ind)
        
        # Track the error for plotting 
        if SHOW_PLOTS:
            if (target_ind + 1) < len(cx):
                print('target index {}'.format(target_ind))
                e_ct, sta, sta_before = calc_error(state, cx, cy, target_ind, sta_before)
                # print('e_ct: {}\nsta: {}\nsta_before: {}'.format(e_ct, sta, sta_before))
                sta_cum += sta
                station.append(sta_cum)
                e_ct_arr.append(e_ct)
            else:
                break

        x0 = [state.x, state.y, state.v, state.yaw]  # current state

        oa, odelta, ox, oy, oyaw, ov = iterative_linear_mpc_control(
            xref, x0, dref, oa, odelta)

        if odelta is not None:
            di, ai = odelta[0], oa[0]

        state = update_state(state, ai, di)
        time = time + DT

        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(time)
        d.append(di)
        a.append(ai)

        if check_goal(state, goal, target_ind, len(cx)):
            print("Goal")
            break

        if SHOW_ANIMATION:  # pragma: no cover
            plt.cla()
            if ox is not None:
                plt.plot(ox, oy, "xr", label="MPC")
            plt.plot(cx, cy, "-r", label="course")
            plt.plot(x, y, "ob", label="trajectory")
            plt.plot(xref[0, :], xref[1, :], "xk", label="xref")
            plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
            plot_car(state.x, state.y, state.yaw, steer=di)
            plt.axis("equal")
            plt.grid(True)
            plt.title("Time[s]:" + str(round(time, 2))
                      + ", speed[m/h]:" + str(round(state.v, 2)))
            plt.pause(0.0001)

    return t, x, y, yaw, v, d, a, station, e_ct_arr


def calc_speed_profile(cx, cy, cyaw, target_speed):

    speed_profile = [target_speed] * len(cx)
    direction = 1.0  # forward

    # Set stop point
    for i in range(len(cx) - 1):
        dx = cx[i + 1] - cx[i]
        dy = cy[i + 1] - cy[i]

        move_direction = math.atan2(dy, dx)

        if dx != 0.0 and dy != 0.0:
            dangle = abs(pi_2_pi(move_direction - cyaw[i]))
            if dangle >= math.pi / 4.0:
                direction = -1.0
            else:
                direction = 1.0

        if direction != 1.0:
            speed_profile[i] = - target_speed
        else:
            speed_profile[i] = target_speed

    speed_profile[-1] = 0.0

    return speed_profile


def smooth_yaw(yaw):

    for i in range(len(yaw) - 1):
        dyaw = yaw[i + 1] - yaw[i]

        while dyaw >= math.pi / 2.0:
            yaw[i + 1] -= math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

        while dyaw <= -math.pi / 2.0:
            yaw[i + 1] += math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

    return yaw

def getLaneChange():
    ax = [0.0, 29.0, 30.0, 40.0, 41.0, 100.0] # longer course
    # ax = [0.0, 19.0, 20.0, 30.0, 31.0, 50.0] # shorter course
    ay = [0.0,  0.0,  0.0,  3.0,  3.0,  3.0]
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
    print(__file__ + " start!!")

    # Chose course by global variable
    if LANE_CHANGE:
        ax, ay, course_name = getLaneChange()
    elif FIGURE_EIGHT:
        ax, ay, course_name = getFigureEight()
    elif ROAD:
        ax, ay, course_name = getRoadPath()

    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(ax, ay, ds=DL)

    sp = calc_speed_profile(cx, cy, cyaw, TARGET_SPEED)

    initial_state = State(x=cx[0], y=0.5, yaw=cyaw[0], v=TARGET_SPEED)

    t, x, y, yaw, v, d, a, station, e_ct_arr = do_simulation(
        cx, cy, cyaw, ck, sp, initial_state, course_name)

    if SHOW_PLOTS:  # pragma: no cover
        # plt.close("all")
        plt.figure(1)
        plt.plot(cx, cy, "-r", label="spline")
        # plt.scatter(cx, cy, "r", s=0.9, label="spline")
        plt.plot(x, y, "-g", label="tracking")
        plt.grid(True)
        plt.axis("equal")
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.legend()

        # plt.subplots()
        # plt.plot(t, v, "-r", label="speed")
        # plt.grid(True)
        # plt.xlabel("Time [s]")
        # plt.ylabel("Speed [kmh]")

        # Plot relevant errors
        plt.figure(0)
        plt.plot(station, e_ct_arr, c=COLOR, label="Q={}".format(Q[0, 0]))
        plt.xlabel("Station[m]")
        plt.ylabel("Cross-Track Error[m]")
        plt.title("MPC at {}m/s on {}".format(TARGET_SPEED, course_name))
        plt.grid(True)

if __name__ == '__main__':
    NX = 4  # x = x, y, v, yaw
    NU = 2  # a = [accel, steer]
    T = 5   # horizon length - should be above 5

    # mpc parameters
    R = np.diag([0.01, 0.01])  # input cost matrix on [accel, steer]
    Rd = np.diag([0.01, 1.0])  # input difference cost matrix
    Q = np.diag([1.0, 1.0, 0.5, 0.5])  # state cost matrix on [x, y, v, yaw]
    # Q = np.diag([0.1, 0.1, 0.5, 0.5]) # potential improvement?
    Qf = Q  # state final matrix
    GOAL_DIS = 1.5  # goal distance
    STOP_SPEED = 0.5 / 3.6  # stop speed
    MAX_TIME = 500.0  # max simulation time

    # iterative paramter
    MAX_ITER = 3  # Max iteration
    DU_TH = 0.1  # iteration finish param
    N_IND_SEARCH = 10  # Search index number

    # Vehicle parameters
    LENGTH = 4.5  # [m]
    WIDTH = 2.0  # [m]
    BACKTOWHEEL = 1.0  # [m]
    WHEEL_LEN = 0.3  # [m]
    WHEEL_WIDTH = 0.2  # [m]
    TREAD = 0.7  # [m]
    WB = 2.5  # [m]
    MAX_STEER = np.deg2rad(45.0)   # maximum steering angle [rad]
    MAX_DSTEER = np.deg2rad(30.0)  # maximum steering speed [rad/s]
    MAX_SPEED = 30.0  # maximum speed [m/s]
    MIN_SPEED = -5   # minimum speed [m/s]
    MAX_ACCEL = 1.0  # maximum accel [m/ss]

    ### CHOOSE RELEVANT PARAMETERS HERE! ###
    TARGET_SPEED = 15.0 # [m/s]
    DL = 2.0 # [m] course tick (DL > 2 if DT < 4)
    DT = 0.005 # [s] time tick -> should be below 0.1

    ### CHOOSE COURSE HERE! ###
    LANE_CHANGE = False
    FIGURE_EIGHT = True
    ROAD = False

    SHOW_ANIMATION = False # Turn on animation
    SHOW_PLOTS = True # Turn on course and error plots

    ### CHOOSE PARAMETERS HERE! ###
    COLOR = 'black'
    main()

    # COLOR = 'blue'
    # main()

    # COLOR = 'cyan'
    # main()

    # COLOR = 'limegreen'
    # main()

    # COLOR = 'pink'
    # main()

    # COLOR = 'red'
    # main()

    plt.legend(loc="best")

    # Save out plots as png files
    if LANE_CHANGE:
        plt.savefig('images/MPC_lane_change_{}.png'.format(int(TARGET_SPEED)))
    elif FIGURE_EIGHT:
        plt.savefig('images/MPC_figure_eight_{}.png'.format(int(TARGET_SPEED)))
    elif ROAD:
        plt.savefig('images/MPC_road_{}.png'.format(int(TARGET_SPEED)))

    plt.show()