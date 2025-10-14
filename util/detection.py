import cv2


def detect_video(vid_path, model):
    # Open the video using OpenCV
    video_capture = cv2.VideoCapture(vid_path)
    screen_width = 1280
    screen_height = 720

    ret, scale = True, 0.2
    waitKey = 2
    while ret: 
        ret, frame = video_capture.read()  # Read a frame   
        results2 = model.track(frame,persist=True,  verbose=False)
        frame_ = results2[0].plot()
        
        frame_ = cv2.resize(frame_, (int(frame_.shape[1]*scale),int(frame_.shape[0]*scale)))
        cv2.imshow("Video Player", frame_)  # Display the frame
        
        if cv2.waitKey(waitKey) & 0xFF == ord('q'):
            break

        if  cv2.waitKey(waitKey) & 0xFF == ord('p'):
            print("P pressed")
            if waitKey == 0:
                waitKey = 2
            else:
                waitKey = 0
    video_capture.release()
    cv2.destroyAllWindows()



def detect_image(im_path, model):
        # Open the video using OpenCV
    img = cv2.imread(im_path)
    result = model.predict(img,persist=True,  verbose=False)
    
    res_img = result[0].plot()
    cv2.imshow("Image", res_img)  # Display the frame
    cv2.waitKey(0)

    img.release()
    cv2.destroyAllWindows()

