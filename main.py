import cv2
import numpy as np

class CurrencyDetector:
    def __init__(self, training_image_path):
        self.orb = cv2.ORB_create(nfeatures=2000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.training_img = cv2.imread(training_image_path, cv2.IMREAD_GRAYSCALE)
        
        if self.training_img is None:
            raise ValueError("Training image not found. Check the path.")
            
        self.kp_train, self.des_train = self.orb.detectAndCompute(self.training_img, None)

    def validate(self, query_image_path, threshold=50):
        query_img = cv2.imread(query_image_path, cv2.IMREAD_GRAYSCALE)
        if query_img is None:
            return "Error: Image not found"

        kp_query, des_query = self.orb.detectAndCompute(query_img, None)
        
        if des_query is None:
            return "No features detected"

        matches = self.matcher.match(self.des_train, des_query)
        matches = sorted(matches, key=lambda x: x.distance)

        good_matches = [m for m in matches if m.distance < 30]
        match_count = len(good_matches)

        if match_count > threshold:
            return f"Authentic: {match_count} points matched"
        else:
            return f"Counterfeit/Suspicious: Only {match_count} points matched"

if __name__ == "__main__":
    # Example usage
    detector = CurrencyDetector('dataset/original_note.jpg')
    result = detector.validate('uploads/test_note.jpg')
    print(result)
