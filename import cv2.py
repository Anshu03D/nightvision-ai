import cv2
import numpy as np

# Create a black image as a test
img = np.zeros((512, 512, 3), np.uint8)
cv2.putText(img, "Mac Setup Ready!", (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

cv2.imshow('Test', img)
cv2.waitKey(0)
cv2.destroyAllWindows()