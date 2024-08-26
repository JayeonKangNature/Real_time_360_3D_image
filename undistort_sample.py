import os
import cv2
import numpy as np
from ocamcamera import OcamCamera


def main():
    current_folder = os.path.dirname(os.path.realpath(__file__))
    left_img_file = os.path.join(r'C:\Users\HyeondoJang\Desktop\HyeondoJang\graduate\POSTECH\intern\code\360-SL-Metasurface-main-dodo\rendered\distorted\wo_line\left\left_distorted_img_0.png')
    right_img_file = os.path.join(r'C:\Users\HyeondoJang\Desktop\HyeondoJang\graduate\POSTECH\intern\code\360-SL-Metasurface-main-dodo\rendered\distorted\wo_line\right\right_distorted_img_0.png')
    ocam_file = os.path.join(r'C:\Users\HyeondoJang\Desktop\HyeondoJang\graduate\POSTECH\intern\code\360-SL-Metasurface-main-dodo\calib_results.txt')
    left_img = cv2.imread(left_img_file)
    right_img = cv2.imread(right_img_file)
    ocam = OcamCamera(ocam_file, fov=185)
    print(ocam)

    # valid area
    valid = ocam.valid_area()

    # Perspective projection

    W = 640
    H = 400
    focal_length = 1.8  # 초점 거리 반영
    z = W / 6
    x = [i - W / 2 for i in range(W)]
    y = [j - H / 2 for j in range(H)]
    x_grid, y_grid = np.meshgrid(x, y, sparse=False, indexing='xy')
    point3D = np.stack([x_grid, y_grid, np.full_like(x_grid, z)]).reshape(3, -1)
    mapx, mapy = ocam.world2cam(point3D)
    mapx = mapx.reshape(H, W)
    mapy = mapy.reshape(H, W)
    left_pers_out = cv2.remap(left_img, mapx, mapy, cv2.INTER_LINEAR)
    right_pers_out = cv2.remap(right_img, mapx, mapy, cv2.INTER_LINEAR)

    # Perspective projection에 빨간색 수평선을 추가
    line_color = (0, 0, 255)  # 빨간색 (BGR 형식)
    line_thickness = 2
    num_lines = 7  # 수평선 개수
    for i in range(1, num_lines + 1):
        y = H * i // (num_lines + 1)
        cv2.line(left_pers_out, (0, y), (W, y), line_color, line_thickness)
        cv2.line(right_pers_out, (0, y), (W, y), line_color, line_thickness)

    # Equirectangular projection
    W = 800
    H = 400
    th = np.pi / H
    p = 2 * np.pi / W
    phi = [-np.pi + (i + 0.5) * p for i in range(W)]
    theta = [-np.pi / 2 + (i + 0.5) * th for i in range(H)]
    phi_xy, theta_xy = np.meshgrid(phi, theta, sparse=False, indexing='xy')
    point3D = np.stack(
        [np.sin(phi_xy) * np.cos(theta_xy), np.sin(theta_xy), np.cos(phi_xy) * np.cos(theta_xy)]).reshape(3, -1)
    mapx, mapy = ocam.world2cam(point3D)
    mapx = mapx.reshape(H, W)
    mapy = mapy.reshape(H, W)
    left_equi_out = cv2.remap(left_img, mapx, mapy, cv2.INTER_LINEAR)
    right_equi_out = cv2.remap(right_img, mapx, mapy, cv2.INTER_LINEAR)

    # Equirectangular projection에도 빨간색 수평선을 추가
    for i in range(1, num_lines + 1):
        y = H * i // (num_lines + 1)
        cv2.line(left_equi_out, (0, y), (W, y), line_color, line_thickness)
        cv2.line(right_equi_out, (0, y), (W, y), line_color, line_thickness)

    # 이미지 출력 및 시각화
    left_img = cv2.resize(left_img, (0, 0), fx=0.5, fy=0.5)
    right_img = cv2.resize(right_img, (0, 0), fx=0.5, fy=0.5)
    valid = cv2.resize(valid, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("left src", left_img)
    cv2.imshow("right src", right_img)
    cv2.imshow("valid", valid)
    cv2.imshow("Left Perspective projection with Lines", left_pers_out)
    cv2.imshow("Right Perspective projection with Lines", right_pers_out)
    cv2.imshow("Left Equirectangular projection with Lines", left_equi_out)
    cv2.imshow("Right Equirectangular projection with Lines", right_equi_out)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()