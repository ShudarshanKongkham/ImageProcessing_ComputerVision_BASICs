import cv2
import math

# List to store the points
global topLeft_corner , bottomRight_corner 

# Read the image
source_Img = cv2.imread("boy.jpg", 1)

# resizing the image for better view
resize_factor = 1.5
source_Img = cv2.resize(source_Img, None, fx=resize_factor, fy=resize_factor,
                        interpolation= cv2.INTER_LINEAR)

dummy = source_Img.copy()


# Defining the funciton to draw the rectangle
def drawRectangle( action, corner_x, corner_y, flags, userdata):
    # Referencing global variables
    global topLeft_corner, bottomRight_corner

    # Action to be taken when left mouse button is pressed
    if action == cv2.EVENT_LBUTTONDOWN:
        topLeft_corner = (corner_x,corner_y)
        # # Mark the top left corner 
        # cv2.circle(source_Img, topLeft_corner, 1, (255, 255, 0), cv2.LINE_AA)

    # Action to be taken when left mouse buttion is released
    elif action == cv2.EVENT_LBUTTONUP:
        bottomRight_corner = (corner_x,corner_y)
        
        # Draw the rectangle
        cv2.rectangle(source_Img, topLeft_corner, bottomRight_corner, 
                      (255,255,0), 2, cv2.LINE_AA)
        cv2.imshow("Window", source_Img)

        try :
            # saving the bounding box region
            crop_face = dummy[topLeft_corner[1]:bottomRight_corner[1], 
                              topLeft_corner[0]:bottomRight_corner[0]]
            cv2.imwrite("Face_cropped.png", crop_face)
            print("Cropped Face saved.🤗")
        except:
            print("Face not selected 😶‍🌫️, don't forget to drag and release.😉 \n Clear the points if not needed by pressing C 😁.")

cv2.namedWindow("Window")

# HighGUI functions to be called when mouse event occur
cv2.setMouseCallback("Window", drawRectangle)

k=0
# loop until escape character is pressed
while k!=27:
    cv2.imshow("Window", source_Img)
    # Adding text background
    cv2.rectangle(source_Img,(0,0),(source_Img.shape[1],55),(255,255,255), -1)

    cv2.putText(source_Img, "Choose top-left corner, and drag to bottom-right corner.",
                (7,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0,0,0), 2)
    cv2.putText(source_Img, "Press 'Esc' to exit and 'c' to clear",
                (7,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0,0,0), 2)
    k = cv2.waitKey(20) & 0xFF
    # Another way of clearing
    if k == 99: # if key pressed is c
        source_Img = dummy.copy()
        print("Cleared ✌️")

cv2.destroyAllWindows()

        