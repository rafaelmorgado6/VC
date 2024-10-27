import cv2
capture = cv2.VideoCapture(0)


while (True):
	ret, frame = capture.read()

	#edges = cv2.Canny(frame, 75, 125)
	cv2.imshow('video', cv2.flip(frame, 1))
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break


capture.release()
cv2.destroyAllWindows()