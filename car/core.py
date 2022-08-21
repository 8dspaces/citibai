import RPi.GPIO as GPIO   ## used simulator in MAC
#from RPiSim import GPIO ### simulate dump GPIO
import time
import pyttsx3
import threading

from brain import events

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

class SmartCar(threading.Thread):

    is_start = False

    buzzer = 8

    LED_R = 22
    LED_G = 27 #27
    LED_B = 24

    # 小车电机引脚定义
    IN1 = 20
    IN2 = 21
    IN3 = 19
    IN4 = 26

    ENA = 16
    ENB = 13

    # 小车按键定义 K2?
    key = 8

    # 红外避障引脚定义
    AvoidSensorLeft = 12
    AvoidSensorRight = 17

    # 循迹红外引脚定义
    TrackSensorLeftPin1 = 3  # 定义左边第一个循迹红外传感器引脚为3口
    TrackSensorLeftPin2 = 5  # 定义左边第二个循迹红外传感器引脚为5口
    TrackSensorRightPin1 = 4  # 定义右边第一个循迹红外传感器引脚为4口
    TrackSensorRightPin2 = 18  # 定义右边第二个循迹红外传感器引脚为18口

    # 超声波引脚定义
    EchoPin = 0
    TrigPin = 1

    # 舵机引脚定义
    ServoPin = 23

    pwm_ENA = None
    pwm_ENB = None
    pwm_servo = None

    def __init__(self):

        threading.Thread.__init__(self)

        GPIO.setup(self.buzzer,GPIO.OUT)

        GPIO.setup(self.LED_R, GPIO.OUT)
        GPIO.setup(self.LED_G, GPIO.OUT)
        GPIO.setup(self.LED_B, GPIO.OUT)

        # speaker = Speaker()
        # speaker.start()


    # 电机引脚初始化为输出模式
    # 按键引脚初始化为输入模式
    # 红外避障引脚初始化为输入模式
    def init_car(self):

        GPIO.setup(self.ENA, GPIO.OUT, initial=GPIO.HIGH)
        GPIO.setup(self.IN1, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.IN2, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.ENB, GPIO.OUT, initial=GPIO.HIGH)
        GPIO.setup(self.IN3, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.IN4, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(self.key, GPIO.IN)

        GPIO.setup(self.AvoidSensorLeft, GPIO.IN)
        GPIO.setup(self.AvoidSensorRight, GPIO.IN)

        GPIO.setup(self.TrackSensorLeftPin1, GPIO.IN)
        GPIO.setup(self.TrackSensorLeftPin2, GPIO.IN)
        GPIO.setup(self.TrackSensorRightPin1, GPIO.IN)
        GPIO.setup(self.TrackSensorRightPin2, GPIO.IN)

        # 设置pwm引脚和频率为2000hz
        self.pwm_ENA = GPIO.PWM(self.ENA, 2000)
        self.pwm_ENB = GPIO.PWM(self.ENB, 2000)
        self.pwm_ENA.start(0)
        self.pwm_ENB.start(0)

        GPIO.setup(self.EchoPin, GPIO.IN)
        GPIO.setup(self.TrigPin, GPIO.OUT)
        GPIO.setup(self.ServoPin, GPIO.OUT)

        # 设置舵机的频率和起始占空比
        self.pwm_servo = GPIO.PWM(self.ServoPin, 50)
        self.pwm_servo.start(0)

    def whistle(self):

        GPIO.setup(self.buzzer, GPIO.OUT)
        GPIO.output(self.buzzer, GPIO.LOW)
        time.sleep(0.1)
        GPIO.output(self.buzzer, GPIO.HIGH)
        time.sleep(0.001)
        print("whistle done")

    def blink(self):

        # 显示7种不同的颜色
        GPIO.output(self.LED_R, GPIO.HIGH)
        GPIO.output(self.LED_G, GPIO.LOW)
        GPIO.output(self.LED_B, GPIO.LOW)
        time.sleep(1)
        GPIO.output(self.LED_R, GPIO.LOW)
        GPIO.output(self.LED_G, GPIO.HIGH)
        GPIO.output(self.LED_B, GPIO.LOW)
        time.sleep(1)
        GPIO.output(self.LED_R, GPIO.LOW)
        GPIO.output(self.LED_G, GPIO.LOW)
        GPIO.output(self.LED_B, GPIO.HIGH)
        time.sleep(1)
        GPIO.output(self.LED_R, GPIO.HIGH)
        GPIO.output(self.LED_G, GPIO.HIGH)
        GPIO.output(self.LED_B, GPIO.LOW)
        time.sleep(1)
        GPIO.output(self.LED_R, GPIO.HIGH)
        GPIO.output(self.LED_G, GPIO.LOW)
        GPIO.output(self.LED_B, GPIO.HIGH)
        time.sleep(1)
        GPIO.output(self.LED_R, GPIO.LOW)
        GPIO.output(self.LED_G, GPIO.HIGH)
        GPIO.output(self.LED_B, GPIO.HIGH)
        time.sleep(1)
        GPIO.output(self.LED_R, GPIO.LOW)
        GPIO.output(self.LED_G, GPIO.LOW)
        GPIO.output(self.LED_B, GPIO.LOW)
        time.sleep(1)
        print("blink done")

    # 小车前进
    def forward(self):
        GPIO.output(self.IN1, GPIO.HIGH)
        GPIO.output(self.IN2, GPIO.LOW)
        GPIO.output(self.IN3, GPIO.HIGH)
        GPIO.output(self.IN4, GPIO.LOW)
        self.pwm_ENA.ChangeDutyCycle(100)
        self.pwm_ENB.ChangeDutyCycle(100)

    # 小车后退
    def back(self):
        GPIO.output(self.IN1, GPIO.LOW)
        GPIO.output(self.IN2, GPIO.HIGH)
        GPIO.output(self.IN3, GPIO.LOW)
        GPIO.output(self.IN4, GPIO.HIGH)
        self.pwm_ENA.ChangeDutyCycle(80)
        self.pwm_ENB.ChangeDutyCycle(80)

    # 小车左转
    def left(self):
        GPIO.output(self.IN1, GPIO.LOW)
        GPIO.output(self.IN2, GPIO.LOW)
        GPIO.output(self.IN3, GPIO.HIGH)
        GPIO.output(self.IN4, GPIO.LOW)
        self.pwm_ENA.ChangeDutyCycle(0)
        self.pwm_ENB.ChangeDutyCycle(80)

    # 小车右转
    def right(self):
        GPIO.output(self.IN1, GPIO.HIGH)
        GPIO.output(self.IN2, GPIO.LOW)
        GPIO.output(self.IN3, GPIO.LOW)
        GPIO.output(self.IN4, GPIO.LOW)
        self.pwm_ENA.ChangeDutyCycle(80)
        self.pwm_ENB.ChangeDutyCycle(0)

    # 小车原地左转
    def spin_left(self):
        GPIO.output(self.IN1, GPIO.LOW)
        GPIO.output(self.IN2, GPIO.HIGH)
        GPIO.output(self.IN3, GPIO.HIGH)
        GPIO.output(self.IN4, GPIO.LOW)
        self.pwm_ENA.ChangeDutyCycle(80)
        self.pwm_ENB.ChangeDutyCycle(80)

    # 小车原地右转
    def spin_right(self):
        GPIO.output(self.IN1, GPIO.HIGH)
        GPIO.output(self.IN2, GPIO.LOW)
        GPIO.output(self.IN3, GPIO.LOW)
        GPIO.output(self.IN4, GPIO.HIGH)
        self.pwm_ENA.ChangeDutyCycle(80)
        self.pwm_ENB.ChangeDutyCycle(80)



    # 按键检测
    def key_scan(self):
        while GPIO.input(self.key):
            pass
        while not GPIO.input(self.key):
            time.sleep(0.01)
            if not GPIO.input(self.key):
                time.sleep(0.01)
                while not GPIO.input(self.key):
                    pass

    # 红外避障
    def infrared_avoid(self):
        # 延时2s
        time.sleep(2)

        # try/except语句用来检测try语句块中的错误，从而让except语句捕获异常信息并处理。
        try:
            self.init_car()
            self.key_scan()
            while True:
                # 遇到障碍物,红外避障模块的指示灯亮,端口电平为LOW
                # 未遇到障碍物,红外避障模块的指示灯灭,端口电平为HIGH
                LeftSensorValue = GPIO.input(self.AvoidSensorLeft)
                RightSensorValue = GPIO.input(self.AvoidSensorRight)

                if LeftSensorValue == True and RightSensorValue == True:
                    self.run()  # 当两侧均未检测到障碍物时调用前进函数
                elif LeftSensorValue == True and RightSensorValue == False:
                    self.spin_left()  # 右边探测到有障碍物，有信号返回，原地向左转
                    time.sleep(0.002)
                elif RightSensorValue == True and LeftSensorValue == False:
                    self.spin_right()  # 左边探测到有障碍物，有信号返回，原地向右转
                    time.sleep(0.002)
                elif RightSensorValue == False and LeftSensorValue == False:
                    self.spin_right()  # 当两侧均检测到障碍物时调用固定方向的避障(原地右转)
                    time.sleep(0.002)

        except KeyboardInterrupt:
            pass

    # 寻迹
    def tracking(self):

        # 延时2s
        time.sleep(2)

        # try/except语句用来检测try语句块中的错误，
        # 从而让except语句捕获异常信息并处理。
        try:
            self.init_car()
            self.key_scan()
            while True:
                # 检测到黑线时循迹模块相应的指示灯亮，端口电平为LOW
                # 未检测到黑线时循迹模块相应的指示灯灭，端口电平为HIGH
                TrackSensorLeftValue1 = GPIO.input(self.TrackSensorLeftPin1)
                TrackSensorLeftValue2 = GPIO.input(self.TrackSensorLeftPin2)
                TrackSensorRightValue1 = GPIO.input(self.TrackSensorRightPin1)
                TrackSensorRightValue2 = GPIO.input(self.TrackSensorRightPin2)

                # 四路循迹引脚电平状态
                # 0 0 X 0
                # 1 0 X 0
                # 0 1 X 0
                # 以上6种电平状态时小车原地右转
                # 处理右锐角和右直角的转动
                if (TrackSensorLeftValue1 == False or TrackSensorLeftValue2 == False) \
                        and TrackSensorRightValue2 == False:

                    self.speed_spin_right(100, 100)
                    time.sleep(0.08)

                # 四路循迹引脚电平状态
                # 0 X 0 0
                # 0 X 0 1
                # 0 X 1 0
                # 处理左锐角和左直角的转动
                elif TrackSensorLeftValue1 == False and (
                        TrackSensorRightValue1 == False or TrackSensorRightValue2 == False):
                    self.speed_spin_left(100, 100)
                    time.sleep(0.08)

                # 0 X X X
                # 最左边检测到
                elif TrackSensorLeftValue1 == False:
                    self.speed_spin_left(80, 80)

                # X X X 0
                # 最右边检测到
                elif TrackSensorRightValue2 == False:
                    self.speed_spin_right(80, 80)

                # 四路循迹引脚电平状态
                # X 0 1 X
                # 处理左小弯
                elif TrackSensorLeftValue2 == False and TrackSensorRightValue1 == True:
                    self.speed_left(0, 90)

                # 四路循迹引脚电平状态
                # X 1 0 X
                # 处理右小弯
                elif TrackSensorLeftValue2 == True and TrackSensorRightValue1 == False:
                    self.speed_right(90, 0)

                # 四路循迹引脚电平状态
                # X 0 0 X
                # 处理直线
                elif TrackSensorLeftValue2 == False and TrackSensorRightValue1 == False:
                    self.run(100, 100)

                # 当为1 1 1 1时小车保持上一个小车运行状态

        except KeyboardInterrupt:
            pass


    ### from 超声波避障
    # 小车前进
    def speed_run(self, leftspeed, rightspeed):
        GPIO.output(self.IN1, GPIO.HIGH)
        GPIO.output(self.IN2, GPIO.LOW)
        GPIO.output(self.IN3, GPIO.HIGH)
        GPIO.output(self.IN4, GPIO.LOW)
        self.pwm_ENA.ChangeDutyCycle(leftspeed)
        self.pwm_ENB.ChangeDutyCycle(rightspeed)

    # 小车后退
    def speed_back(self, leftspeed, rightspeed):
        GPIO.output(self.IN1, GPIO.LOW)
        GPIO.output(self.IN2, GPIO.HIGH)
        GPIO.output(self.IN3, GPIO.LOW)
        GPIO.output(self.IN4, GPIO.HIGH)
        self.pwm_ENA.ChangeDutyCycle(leftspeed)
        self.pwm_ENB.ChangeDutyCycle(rightspeed)

    # 小车左转
    def speed_left(self,leftspeed, rightspeed):
        GPIO.output(self.IN1, GPIO.LOW)
        GPIO.output(self.IN2, GPIO.LOW)
        GPIO.output(self.IN3, GPIO.HIGH)
        GPIO.output(self.IN4, GPIO.LOW)
        self.pwm_ENA.ChangeDutyCycle(leftspeed)
        self.pwm_ENB.ChangeDutyCycle(rightspeed)

    # 小车右转
    def speed_right(self,leftspeed, rightspeed):
        GPIO.output(self.IN1, GPIO.HIGH)
        GPIO.output(self.IN2, GPIO.LOW)
        GPIO.output(self.IN3, GPIO.LOW)
        GPIO.output(self.IN4, GPIO.LOW)
        self.pwm_ENA.ChangeDutyCycle(leftspeed)
        self.pwm_ENB.ChangeDutyCycle(rightspeed)

    # 小车原地左转
    def speed_spin_left(self, leftspeed, rightspeed):
        GPIO.output(self.IN1, GPIO.LOW)
        GPIO.output(self.IN2, GPIO.HIGH)
        GPIO.output(self.IN3, GPIO.HIGH)
        GPIO.output(self.IN4, GPIO.LOW)
        self.pwm_ENA.ChangeDutyCycle(leftspeed)
        self.pwm_ENB.ChangeDutyCycle(rightspeed)

    # 小车原地右转
    def speed_spin_right(self, leftspeed, rightspeed):
        GPIO.output(self.IN1, GPIO.HIGH)
        GPIO.output(self.IN2, GPIO.LOW)
        GPIO.output(self.IN3, GPIO.LOW)
        GPIO.output(self.IN4, GPIO.HIGH)
        self.pwm_ENA.ChangeDutyCycle(leftspeed)
        self.pwm_ENB.ChangeDutyCycle(rightspeed)


    # 舵机旋转到指定角度
    def servo_appointed_detection(self, pos):
        for i in range(18):
            self.pwm_servo.ChangeDutyCycle(2.5 + 10 * pos / 180)

    # 舵机旋转超声波测距避障，led根据车的状态显示相应的颜色
    def servo_color_carstate(self):

        # 开红灯
        GPIO.output(self.LED_R, GPIO.HIGH)
        GPIO.output(self.LED_G, GPIO.LOW)
        GPIO.output(self.LED_B, GPIO.LOW)
        self.speed_back(20, 20)
        time.sleep(0.08)
        self.brake()

        # 舵机旋转到0度，即右侧，测距
        self.servo_appointed_detection(0)
        time.sleep(0.8)
        right_distance = self.Distance_test()

        # 舵机旋转到180度，即左侧，测距
        self.servo_appointed_detection(180)
        time.sleep(0.8)
        left_distance = self.Distance_test()

        # 舵机旋转到90度，即前方，测距
        self.servo_appointed_detection(90)
        time.sleep(0.8)
        front_distance = self.Distance_test()

        if left_distance < 30 and right_distance < 30 and front_distance < 30:
            # 亮品红色，掉头
            GPIO.output(self.LED_R, GPIO.HIGH)
            GPIO.output(self.LED_G, GPIO.LOW)
            GPIO.output(self.LED_B, GPIO.HIGH)
            self.speed_spin_right(85, 85)
            time.sleep(0.58)
        elif left_distance >= right_distance:
            # 亮蓝色
            GPIO.output(self.LED_R, GPIO.LOW)
            GPIO.output(self.LED_G, GPIO.LOW)
            GPIO.output(self.LED_B, GPIO.HIGH)
            self.speed_spin_left(85, 85)
            time.sleep(0.28)
        elif left_distance <= right_distance:
            # 亮品红色，向右转
            GPIO.output(self.LED_R, GPIO.HIGH)
            GPIO.output(self.LED_G, GPIO.LOW)
            GPIO.output(self.LED_B, GPIO.HIGH)
            self.speed_spin_right(85, 85)
            time.sleep(0.28)

    # 超声波函数
    def Distance_test(self):
        GPIO.output(self.TrigPin, GPIO.HIGH)
        time.sleep(0.000015)
        GPIO.output(self.TrigPin, GPIO.LOW)
        while not GPIO.input(self.EchoPin):
            pass
        t1 = time.time()
        while GPIO.input(self.EchoPin):
            pass
        t2 = time.time()

        distance = ((t2 - t1) * 340 / 2) * 100
        print("distance is {}".format(distance))
        time.sleep(0.01)
        return distance

    def avoid_ultrasonic(self):
        # 延时2s
        time.sleep(2)

        # try/except语句用来检测try语句块中的错误，
        # 从而让except语句捕获异常信息并处理。
        try:
            self.init_car()
            self.key_scan()
            while True:
                distance = self.Distance_test()
                if distance > 50:
                    # 遇到障碍物,红外避障模块的指示灯亮,端口电平为LOW
                    # 未遇到障碍物,红外避障模块的指示灯灭,端口电平为HIGH
                    left_sensor_value = GPIO.input(self.AvoidSensorLeft)
                    right_sensor_value = GPIO.input(self.AvoidSensorRight)

                    if left_sensor_value == True and right_sensor_value == True:
                        self.speed_run(100, 100)  # 当两侧均未检测到障碍物时调用前进函数
                    elif left_sensor_value == True and right_sensor_value == False:
                        self.speed_spin_left(85, 85)  # 右边探测到有障碍物，有信号返回，原地向左转
                        time.sleep(0.002)
                    elif right_sensor_value == True and left_sensor_value == False:
                        self.speed_spin_right(85, 85)  # 左边探测到有障碍物，有信号返回，原地向右转
                        time.sleep(0.002)
                    elif right_sensor_value == False and left_sensor_value == False:
                        self.speed_spin_right(85, 85)  # 当两侧均检测到障碍物时调用固定方向的避障(原地右转)
                        time.sleep(0.002)
                        self.speed_run(100, 100)
                        GPIO.output(self.LED_R, GPIO.LOW)
                        GPIO.output(self.LED_G, GPIO.HIGH)
                        GPIO.output(self.LED_B, GPIO.LOW)
                elif 30 <= distance <= 50:
                    # 遇到障碍物,红外避障模块的指示灯亮,端口电平为LOW
                    # 未遇到障碍物,红外避障模块的指示灯灭,端口电平为HIGH
                    left_sensor_value = GPIO.input(self.AvoidSensorLeft)
                    right_sensor_value = GPIO.input(self.AvoidSensorRight)

                    if left_sensor_value is True and right_sensor_value is True:
                        self.speed_run(100, 100)  # 当两侧均未检测到障碍物时调用前进函数
                    elif left_sensor_value is True and right_sensor_value is False:
                        self.speed_spin_left(85, 85)  # 右边探测到有障碍物，有信号返回，原地向左转
                        time.sleep(0.002)
                    elif right_sensor_value is True and left_sensor_value is False:
                        self.speed_spin_right(85, 85)  # 左边探测到有障碍物，有信号返回，原地向右转
                        time.sleep(0.002)
                    elif right_sensor_value is False and left_sensor_value is False:
                        self.speed_spin_right(85, 85)  # 当两侧均检测到障碍物时调用固定方向的避障(原地右转)
                        time.sleep(0.002)
                        self.speed_run(60, 60)
                elif distance < 30:
                    self.servo_color_carstate()

        except KeyboardInterrupt:
            pass

    ### 默认的启动Car的方式
    def start_car(self):

        if not self.is_start:
            self.init_car()
            self.avoid_ultrasonic()
            self.is_start = True
        else:
            self.reset()
            self.init_car()
            self.avoid_ultrasonic()
            self.is_start = True

    # 小车停止
    def brake(self):
        GPIO.output(self.IN1, GPIO.LOW)
        GPIO.output(self.IN2, GPIO.LOW)
        GPIO.output(self.IN3, GPIO.LOW)
        GPIO.output(self.IN4, GPIO.LOW)
        print("reset brake")

    def reset(self):
        if self.pwm_ENA:
            self.pwm_ENA.stop()
            self.pwm_ENB.stop()
        GPIO.cleanup()
        print("reset done")

    def hello(self):
        self.speak("你在干嘛")
        print("hello==========")

    def run(self):

        events.speaker_queue.put("begin")

        #TODO: add logic of car
        while True:
            self.whistle()
            time.sleep(2)
