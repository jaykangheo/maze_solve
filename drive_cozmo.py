import cozmo
from cozmo.robot import MAX_LIFT_HEIGHT_MM, MAX_LIFT_HEIGHT
from cozmo.util import degrees, distance_mm, speed_mmps
import cv2
import numpy as np
import time
from simple_maze import return_path


def solve_hard_maze(robot: cozmo.robot.Robot):
    path = return_path()
    robot.camera.image_stream_enabled = True
    # robot.set_head_angle(degrees(-25)).wait_for_completed()
    robot.set_head_angle(degrees(-25)).wait_for_completed()
    robot.set_lift_height(MAX_LIFT_HEIGHT_MM, in_parallel=True).wait_for_completed()
    # robot.set_head_angle(cozmo.robot.MIN_HEAD_ANGLE).wait_for_completed()

    # robot.turn_in_place(degrees(180)).wait_for_completed()
    count = 0
    while True:
        curim = robot.world.latest_image
        if curim is None:
            time.sleep(0.1)
            curim = robot.world.latest_image

        image = np.array(curim.raw_image).astype(np.uint8)

        # print(image.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # print(image.get(cv2.CAP_PROP_FRAME_WIDTH))

        # image = image[150:300, 60:300]
        straight = image[120:220,107:240]
        hsv = cv2.cvtColor(straight, cv2.COLOR_RGB2HSV)
        # Define range of white color in HSV
        lower_white = np.array([0, 0, 212])
        upper_white = np.array([131, 255, 255])
        # Threshold the HSV image
        mask = cv2.inRange(hsv, lower_white, upper_white)
        # Remove noise
        kernel_erode = np.ones((4, 4), np.uint8)
        eroded_mask = cv2.erode(mask, kernel_erode, iterations=1)
        kernel_dilate = np.ones((6, 6), np.uint8)
        dilated_mask = cv2.dilate(eroded_mask, kernel_dilate, iterations=1)
        # Find the different contours
        im2, contours, hierarchy = cv2.findContours(dilated_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Sort by area (keep only the biggest one)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

        # left roi
        left = image[180:210, 10:40]
        hsv = cv2.cvtColor(left, cv2.COLOR_RGB2HSV)
        # Define range of white color in HSV
        lower_white = np.array([0, 0, 212])
        upper_white = np.array([131, 255, 255])
        # Threshold the HSV image
        left_mask = cv2.inRange(hsv, lower_white, upper_white)
        # Remove noise
        kernel_erode = np.ones((4, 4), np.uint8)
        eroded_mask = cv2.erode(left_mask, kernel_erode, iterations=1)
        kernel_dilate = np.ones((6, 6), np.uint8)
        dilated_mask = cv2.dilate(eroded_mask, kernel_dilate, iterations=1)
        # Find the different contours
        im2, left_contours, hierarchy = cv2.findContours(dilated_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Sort by area (keep only the biggest one)
        left_contours = sorted(left_contours, key=cv2.contourArea, reverse=True)[:1]

        # right roi
        right = image[180:210, 270:300]
        hsv = cv2.cvtColor(right, cv2.COLOR_RGB2HSV)
        # Define range of white color in HSV
        lower_white = np.array([0, 0, 212])
        upper_white = np.array([131, 255, 255])
        # Threshold the HSV image
        right_mask = cv2.inRange(hsv, lower_white, upper_white)
        # Remove noise
        kernel_erode = np.ones((4, 4), np.uint8)
        eroded_mask = cv2.erode(right_mask, kernel_erode, iterations=1)
        kernel_dilate = np.ones((6, 6), np.uint8)
        dilated_mask = cv2.dilate(eroded_mask, kernel_dilate, iterations=1)
        # Find the different contours
        im2, right_contours, hierarchy = cv2.findContours(dilated_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Sort by area (keep only the biggest one)
        right_contours = sorted(right_contours, key=cv2.contourArea, reverse=True)[:1]

        if len(left_contours)>0:
            c = max(left_contours, key=cv2.contourArea)

            M = cv2.moments(c)

            cx = int(M['m10'] / M['m00'])

            cy = int(M['m01'] / M['m00'])

            cv2.line(image, (cx, 0), (cx, 720), (255, 0, 0), 1)

            cv2.line(image, (0, cy), (1280, cy), (255, 0, 0), 1)

            cv2.drawContours(image, contours, -1, (0, 255, 0), 1)

            print('left', cx)
            if cx <19:
                robot.stop_all_motors()

                robot.drive_wheels(50, 50)
                time.sleep(1.3)
                robot.stop_all_motors()
                robot.turn_in_place(degrees(90)).wait_for_completed()
            elif cx >19:
                robot.stop_all_motors()
                robot.turn_in_place(degrees(10)).wait_for_completed()
                robot.drive_straight(distance_mm(10), speed_mmps(100)).wait_for_completed()
                robot.turn_in_place(degrees(-10)).wait_for_completed()

        elif len(right_contours) > 0:
            c = max(right_contours, key=cv2.contourArea)

            M = cv2.moments(c)

            cx = int(M['m10'] / M['m00'])

            cy = int(M['m01'] / M['m00'])

            cv2.line(image, (cx, 0), (cx, 720), (255, 0, 0), 1)

            cv2.line(image, (0, cy), (1280, cy), (255, 0, 0), 1)

            cv2.drawContours(image, contours, -1, (0, 255, 0), 1)

            print('right', cx)
            if cx <16 and cx>4:

                robot.stop_all_motors()

                robot.drive_wheels(50, 50)
                time.sleep(1.3)
                robot.stop_all_motors()
                robot.turn_in_place(degrees(-90)).wait_for_completed()




        elif len(contours) > 0:

            c = max(contours, key=cv2.contourArea)

            M = cv2.moments(c)

            cx = int(M['m10'] / M['m00'])

            cy = int(M['m01'] / M['m00'])

            cv2.line(image, (cx, 0), (cx, 720), (255, 0, 0), 1)

            cv2.line(image, (0, cy), (1280, cy), (255, 0, 0), 1)

            cv2.drawContours(image, contours, -1, (0, 255, 0), 1)

            print('forward', cx)
            # if cx >89:
            #     robot.stop_all_motors()
            #     robot.turn_in_place(degrees(-10)).wait_for_completed()
            #     robot.drive_straight(distance_mm(10), speed_mmps(100)).wait_for_completed()
            #     robot.turn_in_place(degrees(8)).wait_for_completed()
            # elif cx <89:
            #     robot.stop_all_motors()
            #     robot.turn_in_place(degrees(10)).wait_for_completed()
            #     robot.drive_straight(distance_mm(10), speed_mmps(100)).wait_for_completed()
            #     robot.turn_in_place(degrees(-10)).wait_for_completed()


            print("moving forward")
            # robot.drive_straight(distance_mm(30), speed_mmps(100)).wait_for_completed()
            robot.drive_wheels(50, 50)
            # robot.drive_straight(distance_mm(10), speed_mmps(100)).wait_for_completed()


        cv2.imshow("show", left)
        # cv2.waitKey(0)
        if cv2.waitKey(27) == ord('q'):
            break


def solve_simple_maze(robot: cozmo.robot.Robot):
    path = return_path()
    print(path)
    distance = 0
    for i in range(len(path[:-1])):
        turn = 0
        # print(distance)
        # if i == 5:
        #     distance+=50
        #     # robot.drive_straight(distance_mm(50), speed_mmps(100)).wait_for_completed()
        if i ==4:
            distance+=50
        elif i == 10:
            distance += -15
            # robot.drive_straight(distance_mm(20), speed_mmps(100)).wait_for_completed()
        elif i == 12:
            distance += 70
            # robot.drive_straight(distance_mm(60), speed_mmps(100)).wait_for_completed()
        elif i == 14:
            distance += 45
            # robot.drive_straight(distance_mm(50), speed_mmps(100)).wait_for_completed()
        elif i == 17:
            distance += 205
            # robot.drive_straight(distance_mm(225), speed_mmps(100)).wait_for_completed()
        elif i == 20:
            distance += 45
            # robot.drive_straight(distance_mm(25), speed_mmps(100)).wait_for_completed()
        elif i == 22:
            distance += 70
            # robot.drive_straight(distance_mm(50), speed_mmps(100)).wait_for_completed()
        elif i == 27:
            distance += 0
            # robot.drive_straight(distance_mm(25), speed_mmps(100)).wait_for_completed()
        elif i == 29:
            distance += 70
            # robot.drive_straight(distance_mm(60), speed_mmps(100)).wait_for_completed()
        elif i == 38:
            distance += 220
            # robot.drive_straight(distance_mm(220), speed_mmps(100)).wait_for_completed()
        else:
            distance += 42
            # robot.drive_straight(distance_mm(40), speed_mmps(100)).wait_for_completed()

        if path[i] == 2 and (path[i + 1]) == 0:
            turn = -90
            # print('1')
            # print(distance)
            robot.drive_straight(distance_mm(distance), speed_mmps(100)).wait_for_completed()
            robot.turn_in_place(degrees(turn)).wait_for_completed()
            distance = 0

        elif path[i] == 0 and (path[i + 1]) == 3:
            turn = -90
            # print('2')
            # print(distance)
            robot.drive_straight(distance_mm(distance), speed_mmps(100)).wait_for_completed()
            robot.turn_in_place(degrees(turn)).wait_for_completed()
            distance = 0

        elif path[i] == 3 and (path[i + 1]) == 0:
            turn = 90
            # print('3')
            # print(distance)
            robot.drive_straight(distance_mm(distance), speed_mmps(100)).wait_for_completed()
            robot.turn_in_place(degrees(turn)).wait_for_completed()
            distance = 0

        elif path[i] == 0 and (path[i + 1]) == 2:
            turn = 90
            # print('4')
            # print(distance)
            robot.drive_straight(distance_mm(distance), speed_mmps(100)).wait_for_completed()
            robot.turn_in_place(degrees(turn)).wait_for_completed()
            distance = 0

    robot.drive_straight(distance_mm(300), speed_mmps(100)).wait_for_completed()






# cozmo.run_program(solve_simple_maze)

cozmo.run_program(solve_hard_maze)

