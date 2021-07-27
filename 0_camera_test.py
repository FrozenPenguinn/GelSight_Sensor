import cv2

# 58.158319870759286 pix / mm

def gstreamer_pipeline(
        capture_width = 1280,
        capture_height = 720,
        display_width = 1280,
        display_height = 720,
        framerate = 30,
        flip_method = 0,
        ):
    return(
            "nvarguscamerasrc sensor-id=0 wbmode=0 ! "
            "video/x-raw(memory:NVMM), "
            "width=(int)%d, height=(int)%d, "
            "format=(string)NV12, framerate=(fraction)%d/1 !"
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            %(
                capture_width,
                capture_height,
                framerate,
                flip_method,
                display_width,
                display_height,
                )
            )

if __name__ == "__main__":
    # print(gstreamer_pipeline())
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if cap.isOpened():
        window_handle = cv2.namedWindow("CSI Camera",cv2.WINDOW_AUTOSIZE)
        while(cv2.getWindowProperty("CSI Camera", 0) >= 0):
            ret, img = cap.read()
            height, width = img.shape[0:2]
            cv2.imshow("CSI Camera", img)
            if cv2.waitKey(1) == 27:
                cv2.imwrite("./img/length.jpg", img)
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Failed to connect camera")
