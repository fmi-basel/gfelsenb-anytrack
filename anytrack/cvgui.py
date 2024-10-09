import cv2
import numpy as np

class GUI:
    def __init__(self, image, title='preview', mode='scale'):
        self.image = image
        self.image_copy = image.copy()
        self.points = []
        self.line_length = 0
        self.window_name = title
        self.mode = mode
        self.dragged = False

        # Initialize OpenCV window and set the mouse callback
        cv2.namedWindow(self.window_name)
        if self.mode == 'scale':
            cv2.setMouseCallback(self.window_name, self.draw_lines)
        elif self.mode == 'points':
            cv2.setMouseCallback(self.window_name, self.draw_points)

    def calculate_distance(self, pt1, pt2):
        """Calculate the Euclidean distance between two points in pixel space."""
        return int(np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2))

    def draw_lines(self, event, x, y, flags, param):
        """Handle mouse events and draw line with text when two points are selected."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))

            # If two points are selected, draw a line and calculate the distance
            if len(self.points) == 2:
                pt1, pt2 = self.points
                self.line_length = self.calculate_distance(pt1, pt2)

                # Draw the line
                cv2.line(self.image_copy, pt1, pt2, (0, 255, 0), 2)

                # Display the length above the line
                mid_point = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
                cv2.putText(self.image_copy, f"{self.line_length} px", mid_point,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                # Update the display
                cv2.imshow(self.window_name, self.image_copy)

    def draw_points(self, event, x, y, flags, param):
        """Handle mouse events and draw points with text."""
        if event == cv2.EVENT_LBUTTONDOWN:
            print('down')
            cv2.circle(self.image_copy, (x,y), 3, (255, 255, 255), 1)
            # Update the display
            cv2.imshow(self.window_name, self.image_copy)
            self.dragged = True
        if event == cv2.EVENT_MOUSEMOVE and self.dragged:
            self.image_copy = self.image.copy()  # Reset the image
            cv2.circle(self.image_copy, (x,y), 3, (255, 255, 255), 1)
            # Update the display
            cv2.imshow(self.window_name, self.image_copy)
        if event == cv2.EVENT_LBUTTONUP:
            print('up')
            self.dragged = False
            self.points.append((x,y))
            # Update the display
            cv2.imshow(self.window_name, self.image_copy)
        for i, pt in enumerate(self.points):
            cv2.circle(self.image_copy, pt, 3, (255, 255, 0), 1)
            cv2.putText(self.image_copy, f"odor {i+1}: ({pt[0]},{pt[1]})", (10, i*30+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)


    def reset(self):
        """Reset objects."""
        self.points = []
        self.line_length = 0
        self.image_copy = self.image.copy()  # Reset the image
        cv2.imshow(self.window_name, self.image_copy)  # Update the display

    def loop(self):
        """Main loop to display the image and handle key events."""
        while True:
            # Display the image
            cv2.imshow(self.window_name, self.image_copy)

            # Wait for key press
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC key
                # Exit the loop and return the line length
                break
            elif key == 8 or key == 127:  # DEL or Backspace key
                # Reset the line and clear the image
                self.reset()

        # Close the window and return the line length
        cv2.destroyAllWindows()
        if self.mode == 'scale':
            return self.line_length
        elif self.mode == 'points':
            return self.points

# Example usage:
if __name__ == "__main__":
    # Example: Create a simple 8-bit unsigned integer image (grayscale or colored)
    img = np.zeros((500, 500, 3), dtype=np.uint8)

    # Create a GUI object with the image
    gui = GUI(img)

    # Display the image with the line drawing functionality
    length = gui.loop()

    print(f"Final line length in pixels: {length}")
