import pyglet
import numpy as np
pyglet.clock.set_fps_limit(30)


class Viewer(pyglet.window.Window):
    color = {
        'background': [1]*3 + [1]
    }
    fps_display = pyglet.clock.ClockDisplay()
    scale_fac = 200
    bar_thc = 5
    point_l = 1/20
    viewer_xy = (2*scale_fac, 2*scale_fac)

    def __init__(self, arm_info, point_info, n_arms, visible, desired_info):
        width, height = self.viewer_xy
        super(Viewer, self).__init__(width, 
                                     height, 
                                     resizable=False, 
                                     caption='Arm', 
                                     vsync=False, 
                                     visible=visible)  # vsync=False to not use the monitor FPS
        self.set_location(x=80, y=10)
        pyglet.gl.glClearColor(*self.color['background'])

        self.arm_info = arm_info
        self.desired_info = desired_info
        self.point_info = point_info

        self.n_arms = n_arms
        
        self.center_coord = np.array((self.scale_fac, self.scale_fac))
        self.batch = pyglet.graphics.Batch()
        
        # Colors
        c1, c2, c3 = (249, 0, 255)*4, (86, 109, 249)*4, (249, 39, 65)*4
        
        # Objects
        point_box = [0]*8
        self.point = self.batch_add(point_box, c2)
        arm_box = [[0]*8 for i in range(self.n_arms)]
        self.arm = [self.batch_add(arm_box[i], c1) for i in range(self.n_arms)]
        
        if self.desired_info is not None:
            arm_box_des = [[0]*8 for i in range(self.n_arms)]
            self.arm_des = [self.batch_add(arm_box_des[i], c3) for i in range(self.n_arms)]

        self.info = None

    def batch_add(self, v2f, c3b):
        return self.batch.add(4, pyglet.gl.GL_QUADS, None, ('v2f', v2f), ('c3B', c3b))

    def set_vals(self, arm_info, desired_info, info):
        self.arm_info = arm_info.copy()
        self.desired_info = desired_info if desired_info is None else desired_info.copy()
        self.info = info.copy()

    def set_arm_info(self, arm_info):
        self.arm_info = arm_info

    def render(self):
        pyglet.clock.tick()
        self._update_sim()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()
        self.fps_display.draw()
        self.label.draw()

    def rgb_array(self):
        return pyglet.image.get_buffer_manager().get_color_buffer()

    def scale(self, pts):
        return pts * self.scale_fac + self.scale_fac

    def _update_sim(self):
        self._make_goal(self.point_info, self.point_l)
        if self.desired_info is not None:
            self._update_arm(self.desired_info, self.arm_des)
        self._update_arm(self.arm_info, self.arm)


    def _make_goal(self, point_info, point_l):
        # Draw goal
        point_box = np.asarray([
                     point_info[0] - point_l, point_info[1] - point_l,
                     point_info[0] + point_l, point_info[1] - point_l,
                     point_info[0] + point_l, point_info[1] + point_l,
                     point_info[0] - point_l, point_info[1] + point_l])
        self.point.vertices = self.scale(point_box)

    def _update_arm(self, arm_info, arm):
        # Local to global angles
        global_angles = np.cumsum(arm_info[:, 1])
        arm_info[:,2:4] = self.scale(arm_info[:,2:4])

        # Draw arm boxes
        center_coord = self.center_coord
        for i in range(self.n_arms):
            arm_coord = (*(center_coord), *(arm_info[i, 2:4]))
            arm_thick_rad = np.pi / 2 - global_angles[i]
        
            x01 = arm_coord[0] - np.cos(arm_thick_rad) * self.bar_thc
            y01 = arm_coord[1] + np.sin(arm_thick_rad) * self.bar_thc

            x02 = arm_coord[0] + np.cos(arm_thick_rad) * self.bar_thc
            y02 = arm_coord[1] - np.sin(arm_thick_rad) * self.bar_thc

            x11 = arm_coord[2] + np.cos(arm_thick_rad) * self.bar_thc
            y11 = arm_coord[3] - np.sin(arm_thick_rad) * self.bar_thc

            x12 = arm_coord[2] - np.cos(arm_thick_rad) * self.bar_thc
            y12 = arm_coord[3] + np.sin(arm_thick_rad) * self.bar_thc
        
            arm_box = (x01, y01, x02, y02, x11, y11, x12, y12)

            arm[i].vertices = arm_box

            center_coord = arm_info[i, 2:4] 

        #Distance to goal
        self.label = pyglet.text.Label("{:2f}".format(self.info['is_success']),
                                  font_name='Times New Roman',
                                  font_size=36,
                                  x=self.width//2, y=self.height//2,
                                  anchor_x='center', anchor_y='center',
                                  color=(0, 0, 0, 255))

