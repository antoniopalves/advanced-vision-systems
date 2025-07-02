import cv2
 
I = cv2.imread(r"C:\Users\anton\Documents\Erasmus\AVS\mandrill.jpg")
cv2.imshow("Mandrill Image", I)
cv2.waitKey(0)
cv2.imwrite ("m.png",I ) # zapis obrazu do pliku

cv2.destroyAllWindows () # close all windows
