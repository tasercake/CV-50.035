### img = cv2.imread('imgs/SaltAndPepperNoise.jpg', 0)
mymedian = myMedianBlur(img, 5)
median = cv2.medianBlur(img, 5)

# Note that your implementation is NOT necessary to provide 
# the identical output as OpenCV built-in function. However,
# it should visually very similar.
plt.figure(figsize=(16,8))
plt.subplot(121),plt.imshow(median, 'gray')
plt.title('Opencv Median Blur'),plt.xticks([]),plt.yticks([])
plt.subplot(122),plt.imshow(mymedian, 'gray')
plt.title('My Median Blur'),plt.xticks([]),plt.yticks([])
plt.show()
