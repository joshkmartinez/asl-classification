import cv2

def main():
    hand_cascade = cv2.CascadeClassifier('fist.xml')
    
    cap = cv2.VideoCapture(0)

    while True:
        # Read frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break  
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        hands = hand_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        
        for (x, y, w, h) in hands:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        cv2.imshow('Hand Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
