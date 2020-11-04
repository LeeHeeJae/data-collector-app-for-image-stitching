# -*- coding:utf-8 -*-

import tkinter as tk
from tkinter import ttk
import cv2 as cv
import PIL.Image, PIL.ImageTk
import time
import numpy as np
import math
import os

class DroneScanning:

    def __init__(self):
        self.mode = 'image' # 촬영 모드(이미지, 비디오)
        self.s_p_x, self.s_p_y = 0, 0 # 선반의 시작점(매장 내 선반 진열 좌표)
        self.d_xyz_list, self.video_sub_coordinates = [], [] # 드론 선반 스캐닝 좌표(이미지/비디오)
        self.d_x, self.d_y, self.d_z = [], [], []
        self.f_h, self.f_v = 0, 0  # FOV 수평, 수직
        # For panoramic stitching, the ideal set of images will have a reasonable amount of overlap (at least 15–30%)
        # to overcome lens distortion and have enough detectable features.
        self.o_base, self.o_min, self.o_max = 0, 0, 0  # 오버랩 비율(이미지: base, 비디오: min/max)
        self.l_a, self.l_s = 0, 0  # 선반 간 통로 길이, 드론-선반 간 안전 길이
        self.p_vf, self.p_hf = 0, 0  # 파노라마 이미지 생성 비행 횟수(수직/수평)
        self.s_h, self.s_v = 0, 0 # 선반 너비, 선반 높이
        self.extra_region = 0 # extra_region * 2
        self.t_b, self.t_va, self.t_ha = 0.0, 0.0, 0.0
        self.fps = 0 # camera_fps
        self.d_s = 0 # drone_speed
        self.total_variables = [] # 변수 값들이 변화했는지를 확인하는 변수

    def panorama_scanning(self):
        self.total_variables = \
            [self.d_xyz_list, self.f_h, self.f_v, self.o_base, self.o_min, self.o_max,
             self.l_a, self.l_s, self.s_h, self.s_v]

        self.p_vf, self.t_va = self.calculate_number_of_vertical_flight()
        # print('# 드론 Z축 선반 스캐닝 비행 횟수: %f회 / 카메라 수직축 촬영 범위: %f' % (self.p_vf, self.t_va * 2.0))

        self.t_b = self.calculate_base_distance()
        # print('# 드론 선반 기준 거리: ', self.t_b, 'mm')

        self.p_hf, self.t_ha = self.calculate_number_of_horizontal_flight()
        # print('# 드론 Y축 선반 스캐닝 비행 횟수: %f회 / 카메라 수평축 촬영 범위: %f' % (self.p_hf, self.t_ha * 2.0))

        self.d_xyz_list, self.video_sub_coordinates, self.d_x, self.d_y, self.d_z = self.generate_drone_coordinates( )
        # print('드론 선반 스캐닝 좌표: ', self.d_xyz_list)
        # print('비디오 촬영 좌표: ', self.video_sub_coordinates)

    def calculate_number_of_vertical_flight(self):

        # 선반 스캐닝 시 드론 뒷편의 선반과 안전 거리 확인
        max_t_b = self.l_a - self.l_s # 드론이 스캐닝할 선반에서 최대한 떨어질 수 있는 거리
        t_va = max_t_b * np.tan(math.radians(self.f_v))

        if t_va >= self.s_v: # 비행 한번에 전체 선반 스캐닝이 가능한 경우
            p_vf = 1.0
            t_va = self.s_v / (0.5 + (1.0 - self.o_base) * p_vf)  # 20.10.05 수정
        else:
            # p_vf = math.ceil(self.s_v / (2.0 * t_va * (2.0 - self.o_base)) + 1)
            p_vf = math.ceil((self.s_v - (t_va * 0.5)) / (t_va * (1.0 - self.o_base)))  # 20.10.05 수정
            # t_va = self.s_v / (2.0 * (p_vf - 1) * (1.0 - self.o_base))
            t_va = self.s_v / (0.5 + (1.0 - self.o_base) * p_vf) # 20.10.05 수정

        return p_vf, t_va

    def calculate_base_distance(self):
        # 드론과 선반 간 기준거리(직각 삼각형 밑변(base))
        t_b = self.t_va / np.tan(math.radians(self.f_v))
        return t_b

    def calculate_number_of_horizontal_flight(self):
        t_ha = self.t_b * np.tan(math.radians(self.f_h))
        if t_ha >= self.s_h:  # 비행 한번에 전체 선반 스캐닝이 가능한 경우
            p_hf = 1.0
        else:
            # p_hf = (self.s_h - (t_ha * 0.5)) / (t_ha * (1.0 - self.o_base))
            p_hf = math.ceil((self.s_h - (t_ha * 0.5)) / (t_ha * (1.0 - self.o_base)))
        return p_hf, t_ha

    def generate_drone_coordinates(self):
        d_x, d_y, d_z, d_xyz = [], [], [], [] # d_x: 전/후, d_y: 좌/우, d_z: 상/하

        d_x.append(self.t_b) # 전/후

        # 변경된 t_b를 기준으로 p_vf 재계산
        # self.p_vf = math.ceil((self.s_v - (self.t_va * 0.5)) / (self.t_va * (1.0 - self.o_base))) + 1

        if self.mode == 'image': # 이미지 방식
            # 드론 Z축 좌표(선반의 Y축)
            for i in range(self.p_vf):
                if i == 0:
                    # d_z.append(self.t_va * 0.5 - self.extra_region * 0.5 + self.s_p_y)
                    d_z.append(self.t_va * 0.5 + self.s_p_y)
                else:
                    d_z.append(d_z[i-1] + (self.t_va * (1.0 - self.o_base)))
            
            # # 드론 Y축 좌표(마지막은 남은 거리만큼만 이동)(선반의 X축)
            # for i in range(int(math.modf(self.p_hf)[1])):
            #     if i == 0:
            #         d_y.append(self.t_ha * 0.5 + self.s_p_x)
            #     else:
            #         d_y.append(d_y[i-1] + (self.t_ha * (1 - self.o_base)))
            # d_y.append(d_y[-1] + (self.t_ha * (1 - self.o_base) * math.modf(self.p_hf)[0]))


            # # 드론 Y축 좌표(마지막은 남은 거리만큼만 이동)(선반의 X축)
            # for i in range(self.p_hf):
            #     if i == 0:
            #         d_y.append(self.t_ha * 0.5 + self.s_p_x)
            #     else:
            #         d_y.append(d_y[i - 1] + (self.t_ha * (1 - self.o_base)))


            # 드론 Y축 좌표(마지막은 남은 거리만큼만 이동)(선반의 X축)
            last_shooting_pt = 0
            for i in range(self.p_hf):
                if i == 0:
                    d_y.append(self.t_ha * 0.5 + self.s_p_x)
                else:
                    d_y.append(d_y[i - 1] + (self.t_ha * (1 - self.o_base)))
                    last_shooting_pt = d_y[i - 1] + (self.t_ha * (1 - self.o_base))

            # 계산 상 오류로 인해 마지막 촬영 포인트가 나오지 않는 경우가 있음
            d_y.append(last_shooting_pt + (0.5 * self.t_ha * (1 - self.o_base)))


            for i in range(len(d_z)):
                if i % 2 == 1:
                    d_y.reverse()
                for j in range(len(d_y)):
                    d_xyz.append([int(round(d_x[0])), int(round(d_z[i])), int(round(d_y[j]))])
            return d_xyz, [], d_x, d_y, d_z

        elif self.mode == 'video': # 비디오 방식
            print('Development has not been completed')

        else:
            print('wrong selection')

########################################################################################################################

class VideoCapture_usb:

    def __init__(self, video_source=0):
        self.vid = cv.VideoCapture(video_source, cv.CAP_DSHOW)
        if not self.vid.isOpened():
            raise ValueError('unable to open video source', video_source)

        # self.vid.set(3, 1280)
        # self.vid.set(4, 720)
        self.vid.set(3, 640)
        self.vid.set(4, 480)
        self.width = self.vid.get(cv.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                return ret, cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            else:
                return ret, None

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

class App:

    def __init__(self, window, window_title):

        self.vid = VideoCapture_usb(video_source=0) ### VideoCapture class ###
        # self.vid = VideoCapture_rasp(rtsp_usage=True)  ### VideoCapture class ###

        # tkinter 및 camera 관련 파라미터
        self.window = window
        self.tk_extra_w, self.tk_extra_h = 10, 300
        self.window.title(window_title)
        # self.window.geometry("640x400+100+100")
        self.window.geometry(str(int(self.vid.width + self.tk_extra_w)) + 'x' +
                             str(int(self.vid.height + self.tk_extra_h)) + '+50+50')

        self.window.resizable(0, 1)

        self.video_cnt, self.video_enable, self.video_name = 0, False, ''
        self.frame, self.ret, self.out, self.photo = None, None, None, None
        self.mode = 'image'
        self.output_cnt = 0
        self.saved_total_variables = []
        self.shelf_num = 0
        self.shelf_dir, self.shelf_image_dir, self.shelf_video_dir = '', '', ''
        self.f = None
        self.video_time_cnt, self.st_time = 0, 0
        self.y_num, self.x_num = 0, 0

        # entry 파라미터
        self.cam_fps, self.overlap_base, self.overlap_min, self.overlap_max = 15, 0.3, 0.15, 0.4
        ### usb 카메라 사양에 맞는 FoV
        self.usb_fov_h, self.usb_fov_v = np.arctan(1065.0/1000.0)*100.0/2.0, np.arctan(800.0/1000.0)*100.0/2.0
        self.shelf_height, self.shelf_width = 200, 1000
        self.aisle_dist, self.safe_dist, self.extra_region = 200, 30, 20
        self.shelf_name = 'number_name'


        # UI 구성
        ## (1) 프레임 구성
        input_frame = tk.Frame(self.window)
        input_frame.pack(side='top', expand='False', fill='x')

        input_left_frame = \
            tk.Frame(input_frame, highlightbackground="black", highlightthickness=1, padx=5, pady=2)
        input_left_frame.pack(side='left', expand='False', fill='x')

        input_right_frame = tk.Frame(input_frame, highlightbackground="black", highlightthickness=1)
        input_right_frame.pack(side='left', expand='False', fill='both')

        input_right_top_frame = tk.Frame(input_right_frame)
        input_right_top_frame.pack(side='top', expand='False', fill='both')

        input_right_top_left_frame = tk.Frame(input_right_top_frame, highlightbackground="black", highlightthickness=1)
        input_right_top_left_frame.pack(side='left', expand='False', fill='both')

        input_right_top_right_frame = tk.Frame(input_right_top_frame)
        input_right_top_right_frame.pack(side='right', expand='True', fill='both')

        input_right_mid_frame = tk.Frame(input_right_frame, highlightbackground="black", highlightthickness=1)
        input_right_mid_frame.pack(side='top', expand='True', fill='both')

        input_right_bot_frame = tk.Frame(input_right_frame, highlightbackground="black", highlightthickness=1)
        input_right_bot_frame.pack(side='top', expand='True', fill='both')

        output_frame = tk.Frame(self.window, highlightbackground="black", highlightthickness=1)
        output_frame.pack(side='top', expand='False', fill='both')

        output_left_frame = tk.Frame(output_frame)
        output_left_frame.pack(side='left', expand='False', fill='x')

        img = tk.PhotoImage(file='./app_images/drone_xyz_img.gif')
        tk.Label(output_left_frame, image=img).pack(expand='True', fill='both', side='top')

        output_right_frame = tk.Frame(output_frame)
        output_right_frame.pack(side='left', expand='True', fill='x')

        video_frame = tk.Frame(self.window)
        video_frame.pack(side='top', expand='yes', fill='both')

        bot_frame = tk.Frame(self.window, height=50)
        bot_frame.pack(side='top', expand='False', fill='both')

        bot_left_frame = tk.Frame(bot_frame, highlightbackground="black", highlightthickness=1, padx=5, pady=2)
        bot_left_frame.pack(side='left', expand='True', fill='both')

        bot_right_frame = tk.Frame(bot_frame, highlightbackground="black", highlightthickness=1, padx=5, pady=2)
        bot_right_frame.pack(side='left', expand='True', fill='both')

        ## (2) 상단 구성
        height_txt, width_txt, aisle_dist_txt = tk.StringVar(), tk.StringVar(), tk.StringVar()
        safe_dist_txt, fov_hor_txt, fov_ver_txt = tk.StringVar(), tk.StringVar(), tk.StringVar()
        overlap_cent_txt, overlap_min_txt, overlap_max_txt = tk.StringVar(), tk.StringVar(), tk.StringVar()
        cam_fps_txt, extra_region_txt = tk.StringVar(), tk.StringVar()
        shelf_num_txt = tk.StringVar()

        cam_fps_txt.set("카메라 fps: ")
        fov_hor_txt.set("수평 FOV(deg.): ")
        fov_ver_txt.set("수직 FOV(deg.): ")

        overlap_cent_txt.set("기본 오버랩 비율(%): ")
        overlap_min_txt.set("최소 오버랩 비율(%): ")
        overlap_max_txt.set("최대 오버랩 비율(%): ")

        height_txt.set("최대 선반 높이(cm): ")
        width_txt.set("총 선반 너비(cm): ")
        aisle_dist_txt.set("통로 간격(cm): ")

        safe_dist_txt.set("안전 거리(cm): ")
        extra_region_txt.set("추가 영역 크기(cm): ")

        shelf_num_txt.set('선반 이름')

        ### 카메라 fps
        tk.Label(input_left_frame, textvariable=cam_fps_txt).grid(row=0, column=0, sticky='e')
        self.cam_fps_entry = tk.Entry(input_left_frame, width=5)
        self.cam_fps_entry.insert(0, str(self.cam_fps))
        self.cam_fps_entry.grid(row=0, column=1)

        ### 카메라 FOV hor/ver
        tk.Label(input_left_frame, textvariable=fov_hor_txt).grid(row=0, column=2, sticky='e')
        self.fov_hor_entry = tk.Entry(input_left_frame, width=5)
        self.fov_hor_entry.insert(0, str(self.usb_fov_h))
        self.fov_hor_entry.grid(row=0, column=3)

        tk.Label(input_left_frame, textvariable=fov_ver_txt).grid(row=0, column=4, sticky='e')
        self.fov_ver_entry = tk.Entry(input_left_frame, width=5)
        self.fov_ver_entry.insert(0, str(self.usb_fov_v))
        self.fov_ver_entry.grid(row=0, column=5)

        ### 오버랩 비율(기본, 최저, 최대)
        tk.Label(input_left_frame, textvariable=overlap_cent_txt).grid(row=1, column=0, sticky='e')
        self.overlap_base_entry = tk.Entry(input_left_frame, width=5)
        self.overlap_base_entry.insert(0, str(self.overlap_base))
        self.overlap_base_entry.grid(row=1, column=1)

        tk.Label(input_left_frame, textvariable=overlap_min_txt).grid(row=1, column=2, sticky='e')
        self.overlap_min_entry = tk.Entry(input_left_frame, width=5)
        self.overlap_min_entry.insert(0, str(self.overlap_min))
        self.overlap_min_entry.grid(row=1, column=3)

        tk.Label(input_left_frame, textvariable=overlap_max_txt).grid(row=1, column=4, sticky='e')
        self.overlap_max_entry = tk.Entry(input_left_frame, width=5)
        self.overlap_max_entry.insert(0, str(self.overlap_max))
        self.overlap_max_entry.grid(row=1, column=5)

        ### 선반 높이
        tk.Label(input_left_frame, textvariable=height_txt).grid(row=2, column=0, sticky='e')
        self.shelf_height_entry = tk.Entry(input_left_frame, width=5)
        self.shelf_height_entry.insert(0, str(self.shelf_height))
        self.shelf_height_entry.grid(row=2, column=1)

        ### 선반 너비
        tk.Label(input_left_frame, textvariable=width_txt).grid(row=2, column=2, sticky='e')
        self.shelf_width_entry = tk.Entry(input_left_frame, width=5)
        self.shelf_width_entry.insert(0, str(self.shelf_width))
        self.shelf_width_entry.grid(row=2, column=3)

        ### 통로 길이
        tk.Label(input_left_frame, textvariable=aisle_dist_txt).grid(row=2, column=4, sticky='e')
        self.aisle_dist_entry = tk.Entry(input_left_frame, width=5)
        self.aisle_dist_entry.insert(0, str(self.aisle_dist))
        self.aisle_dist_entry.grid(row=2, column=5)

        ### 안전 길이
        tk.Label(input_left_frame, textvariable=safe_dist_txt).grid(row=3, column=0, sticky='e')
        self.safe_dist_entry = tk.Entry(input_left_frame, width=5)
        self.safe_dist_entry.insert(0, str(self.safe_dist))
        self.safe_dist_entry.grid(row=3, column=1)

        ### 여분 길이
        tk.Label(input_left_frame, textvariable=extra_region_txt).grid(row=3, column=2, sticky='e')
        self.extra_region_entry = tk.Entry(input_left_frame, width=5)
        self.extra_region_entry.insert(0, str(self.extra_region))
        self.extra_region_entry.grid(row=3, column=3)

        ### help 버튼
        # tk.Button(input_left_frame, command=self.show_popup, text='HELP', bg='orange').grid(row=3, column=4)

        ### help 버튼
        tk.Button(input_left_frame, command=self.call_param_logs, text='PREV', bg='orange').grid(row=3, column=5)

        ### 콤보 박스
        tk.Label(input_right_top_left_frame, text='촬영 모드').pack(expand='False', fill='both', side ='left')
        self.str_var = tk.StringVar()
        combo = ttk.Combobox(input_right_top_right_frame, textvariable=self.str_var)
        combo['values'] = ('image', 'video')
        combo.pack(expand='True', fill='both', side ='left')
        combo.current(0)

        # ### 선반 이름
        tk.Label(input_right_mid_frame, textvariable=shelf_num_txt).pack(expand='False', fill='both', side='left')
        self.shelf_name_entry = tk.Entry(input_right_mid_frame)
        self.shelf_name_entry.insert(0, self.shelf_name)
        self.shelf_name_entry.pack(expand='True', fill='both', side='left')

        ### 버튼
        tk.Button(input_right_bot_frame,
                  command=lambda:[self.get_values(), self.select_mode(), self.make_directories(), self.save_param_logs()]
                  , text='OK').pack(expand='True', fill='both')

        ### 선반 스캐닝 좌표 출력 label
        scrbar = tk.Scrollbar(output_right_frame)
        scrbar.pack(expand='False', fill='both', side='right')
        self.output_label = tk.Listbox(output_right_frame, yscrollcommand=scrbar.set)
        self.output_label.pack(expand='True', fill='both', side='left')
        scrbar.config(command=self.output_label.yview)

        ## (3) 중단 구성
        self.canvas = tk.Canvas(video_frame, width=self.vid.width, height=self.vid.height)
        self.canvas.grid(row=0, column=0, rowspan=2)

        ## (4) 하단 구성
        tk.Button(bot_left_frame, text="image", command=self.get_image).pack(expand='False', fill='both')
        self.video_bnt = tk.Button(bot_right_frame, text="video", command=self.get_video)
        self.video_bnt.pack(expand='False', fill='both')

        self.delay = 60  # default fps = 15 일 때
        self.update()
        self.window.mainloop()

    def select_mode(self):
        self.mode = self.str_var.get()
        # print(self.mode)

    def get_image(self):
        if self.ret:
            cv.imwrite(self.shelf_image_dir + "/image-" + time.strftime("%Y-%m-%d-%H-%M-%S")+".jpg",
                       cv.cvtColor(self.frame, cv.COLOR_RGB2BGR))

    def get_video(self):
        if self.video_cnt % 2 == 0:
            self.video_enable = True
            fourcc = cv.VideoWriter_fourcc(*'DIVX')
            self.video_name = self.shelf_video_dir + "/video-" + time.strftime("%Y-%m-%d-%H-%M-%S")
            self.out = \
                cv.VideoWriter(self.video_name + ".avi", fourcc, float(self.cam_fps_entry.get()),
                               (int(self.vid.width), int(self.vid.height)))

            self.f = open(self.video_name + '.txt', 'w') # 동영상 촬영 시 특정 프레임을 저장하기 위한 텍스트 파일 생성
            self.f.write('[' + str(self.y_num) + ', ' + str(self.x_num) + ']\n')
            self.st_time = time.time()

            self.video_bnt['text'] = ' stop '
            self.video_bnt['bg'] = 'red'

        else:
            self.video_enable = False
            self.video_bnt['text'] = 'video'
            self.video_bnt['bg'] = 'white smoke'
            self.f.close()

        self.video_cnt += 1

    def keyPressed(self, event):
        self.f.write(str(round(time.time() - self.st_time, 2)) + ' sec. ' + str(self.video_time_cnt) + '-th frame\n')

    def update(self):
        self.ret, self.frame = self.vid.get_frame()
        copy_frame = self.frame.copy()
        len_size = 10
        cv.line(copy_frame, (int(self.vid.width / 2) - len_size, int(self.vid.height / 2)),
                (int(self.vid.width / 2) + len_size, int(self.vid.height / 2)), (255, 0, 0), 3)
        cv.line(copy_frame, (int(self.vid.width / 2), int(self.vid.height / 2) - len_size),
                (int(self.vid.width / 2), int(self.vid.height / 2)  + len_size), (255, 0, 0), 3)

        cv.rectangle(copy_frame, (int(self.vid.width * float(self.overlap_base)),
                                  int(self.vid.height * float(self.overlap_base))),
                     (int(self.vid.width * (1.0 - float(self.overlap_base))),
                      int(self.vid.height * (1.0 - float(self.overlap_base)))), (0, 255, 0), 2)

        cv.rectangle(copy_frame, (int(self.vid.width * float(self.overlap_base - 0.05)),
                                  int(self.vid.height * float(self.overlap_base - 0.05))),
                     (int(self.vid.width * (1.0 - float(self.overlap_base) + 0.05)),
                      int(self.vid.height * (1.0 - float(self.overlap_base) + 0.05))), (255, 0, 0), 3)

        if self.video_enable:
            self.video_time_cnt += 1
            self.out.write(cv.cvtColor(self.frame, cv.COLOR_RGB2BGR))
            self.window.bind('<Key>', self.keyPressed)
            self.window.focus_set()

        if self.ret:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(copy_frame))
            self.canvas.create_image(int(self.tk_extra_w / 2), 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update)

    def get_values(self):

        ds = DroneScanning() ### DroneScanning class ###
        ds.mode = self.mode
        ds.extra_region = int(self.extra_region_entry.get())
        ds.s_h, ds.s_v = int(self.shelf_width_entry.get()), int(self.shelf_height_entry.get())
        ds.s_v += ds.extra_region # 세로축에만 여분의 영역을 추가하여 영상을 획득
        ds.f_h, ds.f_v = float(self.fov_hor_entry.get()), float(self.fov_ver_entry.get())
        ds.o_base = float(self.overlap_base_entry.get())
        ds.o_min, ds.o_max = float(self.overlap_min_entry.get()), float(self.overlap_max_entry.get())
        ds.l_a, ds.l_s = int(self.aisle_dist_entry.get()), int(self.safe_dist_entry.get())
        ds.fps = int(self.cam_fps_entry.get())

        # 이전 log를 활용하기 위해 사용되는 self 변수
        self.overlap_base, self.overlap_min, self.overlap_max = ds.o_base, ds.o_min, ds.o_max
        self.usb_fov_h, self.usb_fov_v = ds.f_h, ds.f_v
        self.shelf_height, self.shelf_width = ds.s_v - ds.extra_region, ds.s_h
        self.aisle_dist, self.safe_dist, self.extra_region = ds.l_a, ds.l_s, ds.extra_region
        self.cam_fps = ds.fps

        self.shelf_name = self.shelf_name_entry.get() # 스캔할 선반의 번호

        ds.panorama_scanning()

        self.y_num, self.x_num = len(ds.d_z), len(ds.d_y)
        self.delay = int(1.0 / ds.fps * 1000) # fps 에 맞게 delay 변경

        if 0 in ds.total_variables[1:]:
            self.output_label.configure(text='0이 아닌 숫자를 입력하세요')
            self.saved_total_variables = ds.total_variables
        else:
            # 입력 변수 값이 변화했을 때만 출력
            if self.saved_total_variables != ds.total_variables:

                self.output_label.delete(0, 'end')

                contents_1 = "■ 스캐닝 비행 횟수(Y축): {} / (X축): {}".format(float(ds.p_vf), float(math.ceil(ds.p_hf)))
                contents_2 = "■ 촬영 범위(Y축): {} / (X축): {}".format(round(ds.t_va), round(ds.t_ha))
                converted_unit_1 = str(int(round(ds.t_b))//100) + 'm ' + str(int(round(ds.t_b)) % 100) + 'cm'
                contents_3 = "■ 드론과 선반 간 거리: {}".format(converted_unit_1)
                contents_4 = "■ 선반 촬영 좌표 "

                self.output_label.insert(0, contents_1)
                self.output_label.insert(1, contents_2)
                self.output_label.insert(2, contents_3)
                self.output_label.itemconfig(2, {'fg': 'red'})
                self.output_label.insert(3, contents_4)

                converted_unit_2 = []
                for i in sorted(ds.d_z):
                    converted_unit_2.append(str(int(round(i))//100) + 'm ' + str(int(round(i))%100) +'cm')
                self.output_label.insert(4, 'Y축: ' + str(converted_unit_2))
                self.output_label.itemconfig(4, {'fg': 'red'})

                scr_idx, div_num = 5, 5
                ds.d_y = [round(i - 52) for i in sorted(ds.d_y)]

                for i in range(0, len(ds.d_y), div_num):
                    tmp = ds.d_y[i:i + div_num]
                    converted_unit_3 = []
                    for j in tmp:
                        converted_unit_3.append(str(int(j) // 100) + 'm ' + str(int(j) % 100) + 'cm')
                    self.output_label.insert(scr_idx, 'X축(' + str(scr_idx - 5) + '): ' + str(converted_unit_3))
                    self.output_label.itemconfig(scr_idx, {'fg': 'blue'})
                    scr_idx += 1

            self.saved_total_variables = ds.total_variables

    def save_param_logs(self):
        param_list = [self.cam_fps, self.overlap_base, self.overlap_min, self.overlap_max,
                      self.usb_fov_h, self.usb_fov_v, self.shelf_height, self.shelf_width,
                      self.aisle_dist, self.safe_dist, self.extra_region, self.shelf_name]
        with open('input_param_logs.txt', 'w') as f:
            for i in param_list:
                f.write(str(i)+'\n')

    def call_param_logs(self):
        with open('input_param_logs.txt', 'r') as f:
            param_vals = f.readlines()

        param_vals = [i.replace('\n', '') for i in param_vals]
        self.cam_fps, self.overlap_base, self.overlap_min, self.overlap_max = \
            int(param_vals[0]), float(param_vals[1]), float(param_vals[2]), float(param_vals[3])
        self.usb_fov_h, self.usb_fov_v, self.shelf_height, self.shelf_width = \
            float(param_vals[4]), float(param_vals[5]), int(param_vals[6]), int(param_vals[7])
        self.aisle_dist, self.safe_dist, self.extra_region, self.shelf_name = \
            int(param_vals[8]), int(param_vals[9]), int(param_vals[10]), param_vals[11]

        # 현재 entry 내용 삭제
        self.cam_fps_entry.delete(0, 'end')
        self.fov_hor_entry.delete(0, 'end')
        self.fov_ver_entry.delete(0, 'end')
        self.overlap_base_entry.delete(0, 'end')
        self.overlap_min_entry.delete(0, 'end')
        self.overlap_max_entry.delete(0, 'end')
        self.shelf_height_entry.delete(0, 'end')
        self.shelf_width_entry.delete(0, 'end')
        self.aisle_dist_entry.delete(0, 'end')
        self.safe_dist_entry.delete(0, 'end')
        self.extra_region_entry.delete(0, 'end')
        self.shelf_name_entry.delete(0, 'end')

        # 이전 log 재적용
        self.cam_fps_entry.insert(0, str(self.cam_fps))
        self.fov_hor_entry.insert(0, str(self.usb_fov_h))
        self.fov_ver_entry.insert(0, str(self.usb_fov_v))
        self.overlap_base_entry.insert(0, str(self.overlap_base))
        self.overlap_min_entry.insert(0, str(self.overlap_min))
        self.overlap_max_entry.insert(0, str(self.overlap_max))
        self.shelf_height_entry.insert(0, str(self.shelf_height))
        self.shelf_width_entry.insert(0, str(self.shelf_width))
        self.aisle_dist_entry.insert(0, str(self.aisle_dist))
        self.safe_dist_entry.insert(0, str(self.safe_dist))
        self.extra_region_entry.insert(0, str(self.extra_region))
        self.shelf_name_entry.insert(0, self.shelf_name)

    def make_directories(self):

        self.shelf_dir = 'shelf_' + self.shelf_name
        check_dir_existed = os.path.isdir(self.shelf_dir)

        if self.shelf_image_dir == '' or self.shelf_video_dir == '':
            self.shelf_image_dir, self.shelf_video_dir = self.shelf_dir + '/images', self.shelf_dir + '/videos'

        if not check_dir_existed:
            os.mkdir(self.shelf_dir)
            print(self.shelf_image_dir)
        if not os.path.isdir(self.shelf_image_dir):
            os.mkdir(self.shelf_image_dir)
        if not os.path.isdir(self.shelf_video_dir):
            os.mkdir(self.shelf_video_dir)


    def show_popup(self):
        help_txt = \
            " ■ '기본 오버랩 비율(%)': 이미지 수집 시 연속된 프레임 사이 겹침 영역의 비율 (default = 0.3) \n" \
            " ■ '최소 오버랩 비율(%)': 비디오 수집 시 연속된 프레임 사이 최소 겹침 영역의 비율 (default = 0.15) \n" \
            " ■ '최대 오버랩 비율(%)': 비디오 수집 시 연속된 프레임 사이 최소 겹침 영역의 비율 (default = 0.4) \n" \
            " ■ '최대 선반 높이(mm)': 연속적으로 진열되어 선반 스캐닝을 수행할 선반들 중 가장 큰 선반의 높이 \n" \
            " ■ '총 선반 너비(mm)': 연속적으로 진열되어 선반 스캐닝을 수행할 선반들의 너비 총 합 \n" \
            " ■ '통로 간격(mm)': 선반 스캐닝을 수행할 선반과 드론 뒷편에 마주보는 선반 사이의 통로 길이 \n" \
            " ■ '안전 거리(mm)': 선반 스캐닝을 수행 시 드론과 드론 뒷편에 마주보는 선반 사이의 길이 (default = 500)\n" \
            " ■ '추가 영역 크기(mm)': 파노라마 이미지 생성 시 모서리 영역의 잘림을 방지하기 위한 버퍼 공간 \n"

        popup_txt = tk.StringVar()
        popup_txt.set(help_txt)
        popup_window = tk.Toplevel(self.window)
        popup_label = tk.Label(popup_window, textvariable=popup_txt, justify='left')
        label_font = ('times', 12)
        popup_label.config(font=label_font)
        popup_label.pack()


if __name__ == '__main__':
    App(tk.Tk(), 'Shelf_Scanning_Drone_Camera_UI')
