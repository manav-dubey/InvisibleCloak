import cv2 
import numpy as np 
 
# Initialize the camera 
cap = cv2.VideoCapture(0) 
background = None  # Background will be captured when the user presses 'b' 
 
if not cap.isOpened(): 
    print("Error: Could not open webcam") 
    exit() 
 
print("Press 'b' to capture the background image.") 
 
while cap.isOpened(): 
    ret, frame = cap.read() 
    if not ret: 
        break 
 
    frame = cv2.flip(frame, 1)  # Flip for consistency 
    cv2.imshow("Press 'b' to capture background", frame) 
 
    key = cv2.waitKey(1) & 0xFF 
    if key == ord('b'):  # Press 'b' to capture background 
        background = frame.copy() 
        print("Background captured! Now hold the light pink cloth.") 
        break 
    elif key == ord('q'):  # Press 'q' to exit 
        cap.release() 
        cv2.destroyAllWindows() 
        exit() 
 
if background is None: 
    print("No background captured. Exiting.") 
    cap.release() 
    cv2.destroyAllWindows() 
    exit() 
 
# Create trackbar window for fine-tuning HSV values
cv2.namedWindow('HSV Controls')
cv2.createTrackbar('H Lower', 'HSV Controls', 160, 179, lambda x: None)
cv2.createTrackbar('H Upper', 'HSV Controls', 180, 179, lambda x: None)
cv2.createTrackbar('S Lower', 'HSV Controls', 10, 255, lambda x: None)
cv2.createTrackbar('S Upper', 'HSV Controls', 60, 255, lambda x: None)
cv2.createTrackbar('V Lower', 'HSV Controls', 150, 255, lambda x: None)
cv2.createTrackbar('V Upper', 'HSV Controls', 255, 255, lambda x: None)

# Setting initial values for secondary HSV range
cv2.createTrackbar('H2 Lower', 'HSV Controls', 0, 179, lambda x: None)
cv2.createTrackbar('H2 Upper', 'HSV Controls', 10, 179, lambda x: None)

while cap.isOpened(): 
    ret, frame = cap.read() 
    if not ret: 
        break 
 
    frame = cv2.flip(frame, 1)  # Flip for consistency 
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
 
    # Get current trackbar positions for HSV values
    h_low = cv2.getTrackbarPos('H Lower', 'HSV Controls')
    h_high = cv2.getTrackbarPos('H Upper', 'HSV Controls')
    s_low = cv2.getTrackbarPos('S Lower', 'HSV Controls')
    s_high = cv2.getTrackbarPos('S Upper', 'HSV Controls')
    v_low = cv2.getTrackbarPos('V Lower', 'HSV Controls')
    v_high = cv2.getTrackbarPos('V Upper', 'HSV Controls')
    
    # Get secondary hue range positions (for handling pink wraparound)
    h2_low = cv2.getTrackbarPos('H2 Lower', 'HSV Controls')
    h2_high = cv2.getTrackbarPos('H2 Upper', 'HSV Controls')
    
    # Primary pink color range - higher end of hue scale (magenta to red)
    lower_pink = np.array([h_low, s_low, v_low])
    upper_pink = np.array([h_high, s_high, v_high])
    
    # Secondary pink color range - lower end of hue scale (some pinks wrap to low values)
    lower_pink2 = np.array([h2_low, s_low, v_low])
    upper_pink2 = np.array([h2_high, s_high, v_high])
 
    # Create masks to detect pink
    mask1 = cv2.inRange(hsv, lower_pink, upper_pink) 
    mask2 = cv2.inRange(hsv, lower_pink2, upper_pink2)
    
    # Combine masks to catch both ranges
    mask = cv2.bitwise_or(mask1, mask2)
 
    # Noise removal
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2) 
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1) 
 
    # Create inverse mask 
    mask_inv = cv2.bitwise_not(mask) 
 
    # Extract frame without pink cloth 
    res1 = cv2.bitwise_and(frame, frame, mask=mask_inv) 
 
    # Replace pink cloth with background 
    res2 = cv2.bitwise_and(background, background, mask=mask) 
 
    # Merge both 
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0) 
 
    # Show the detection mask for debugging
    cv2.imshow("Pink Detection Mask", mask)
    
    # Show result 
    cv2.imshow("Invisible Cloak - Pink", final_output) 
    
    # Display current HSV values
    hsv_vals = f"Pink 1: H:{h_low}-{h_high}, S:{s_low}-{s_high}, V:{v_low}-{v_high} | Pink 2: H:{h2_low}-{h2_high}"
    text_img = np.zeros((50, 600, 3), np.uint8)
    cv2.putText(text_img, hsv_vals, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Current HSV Values", text_img)
 
    # Exit when 'q' is pressed 
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break 
 
cap.release() 
cv2.destroyAllWindows()