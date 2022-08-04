import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections.abc import Iterable

def image_to_matrix(image_file, threshold=False):
    image_src = cv2.imread(image_file)
    grayImage = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)
    blackAndWhiteImage = grayImage
    if threshold is True:
    	(thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 0, 255, cv2.THRESH_BINARY)
    blackAndWhiteImage = np.where(blackAndWhiteImage == 0, -1, blackAndWhiteImage)
    blackAndWhiteImage = np.where(blackAndWhiteImage == 255, 1, blackAndWhiteImage)
    return blackAndWhiteImage

def matrix_to_image(image_file, matrix):
    matrix = np.where(matrix == -1, 0, matrix)
    matrix = np.where(matrix == 1, 255, matrix)
    cv2.imwrite(image_file, matrix)
    #cv2.imshow("image", matrix)
    #cv2.waitKey()

class HopfieldNN:
	def __init__(self, act_func=np.sign, fname=None):
		self.M = None
		self.state = None
		self.new_state = None
		self.act_func = act_func
		if fname is not None: 
			self.load_network(fname)

	@property
	def weights(self):
		return self.M

	def train(self, *train_matrices):
		if self.M is None and len(train_matrices) > 0:
			self.M = np.zeros((train_matrices[0].size, train_matrices[0].size))
		for train_data in train_matrices:
			train_data = train_data.flatten()
			for i in range(train_data.size):
				for j in range(i, train_data.size):
					if i == j:
						self.M[i, j] = 0
						continue
					self.M[i, j] += train_data[i] * train_data[j]
					self.M[j, i] = self.M[i, j]

	def input_data(self, data):
		if data.size != self.M.shape[0]:
			print("{}x{} data with size {}".format(data.size, *data.shape, end=' '))
			print("got instead of {}".format(self.M.shape[0]))
			return
		if self.M is None:
			print("No training data provided")
			return
		self.state = data
		self.new_state = None

	def update_state(self):
		if self.state is None:
			print("No input data provided")
			return
		if self.new_state is not None:
			self.state = self.new_state
		self.new_state = np.matmul(self.M, self.state.flatten()).reshape(self.state.shape)
		#state = self.state.flatten()
		#new_state = np.zeros(state.shape)
		#for i in range(self.M.shape[0]):
		#	for j in range(self.M.shape[1]):
		#		new_state[i] += self.M[i, j] * state[j]
		#new_state = new_state.reshape(self.state.shape)
		#print(np.array_equal(new_state, self.new_state))
		self.new_state = self.act_func(self.new_state).astype(int)
		#print(self.new_state)

	def recall(self, iterations=10):
		for iteration in range(iterations):
			self.update_state()
			if np.array_equal(self.state, self.new_state):
				#print("State did not change, returning new state...")
				#print(iteration)
				break
		else:
			print("State is changing", end=' ')
			print("after {} iterations".format(iterations))
		return self.new_state

	def save_network(self, fname):
		np.savetxt(fname + '.csv', self.M, delimiter=',')

	def load_network(self, fname):
		self.M = np.loadtxt(fname + '.csv', delimiter=',')

def work(mode, folder, train_images, spoiled_images):
	"""
	mode: 0 - train and save; 1 - load and word; 2 - train and work
	"""
	if not isinstance(train_images, Iterable):
		train_images = (train_images, )
	if not isinstance(spoiled_images, Iterable):
		spoiled_images = (spoiled_images, )
	
	h = HopfieldNN()
	if mode in (0, 2):
		for fname in train_images:
			train_data = image_to_matrix("{}/{}.png".format(folder, fname))
			h.train(train_data)
		if mode == 0:
			h.save_network('_'.join(train_images))
	if mode in (1, 2):
		if mode == 1:
			h.load_network('_'.join(train_images))
		for fname in spoiled_images:
			spoiled_data = image_to_matrix("{}/{} - Copy.png".format(folder, fname))
			h.input_data(spoiled_data)
			result_data = h.recall()
			matrix_to_image("{}/{}_result.png".format(folder, fname), result_data)

def visualize(folder, images, n_images):
	fig = plt.figure(figsize=(10, 8))
	axes1 = []
	axes2 = []
	axes3 = []
	count = 1
	for i in range(len(images)):
		fname = "{}.png".format(images[i])
		axes1.append(fig.add_subplot(3, len(images), count))
		img = cv2.imread(folder + '/' + fname)
		axes1[-1].imshow(img)
		axes1[-1].set_title(fname)
		axes1[-1].set_xticks([])
		axes1[-1].set_yticks([])
		count += 1
	count += len(n_images) - len(images) if len(n_images) > len(images) else 0
	for i in range(len(n_images)):
		fname = "{} - Copy.png".format(n_images[i])
		axes2.append(fig.add_subplot(3, len(n_images), count))
		img = cv2.imread(folder + '/' + fname)
		axes2[-1].imshow(img)
		axes2[-1].set_title(fname)
		axes2[-1].set_xticks([])
		axes2[-1].set_yticks([])
		count += 1
	for i in range(len(n_images)):
		fname = "{}_result.png".format(n_images[i])
		axes3.append(fig.add_subplot(3, len(n_images), count))
		img = cv2.imread(folder + '/' + fname)
		axes3[-1].imshow(img)
		axes3[-1].set_title(fname)
		axes3[-1].set_xticks([])
		axes3[-1].set_yticks([])
		count += 1
	plt.show()


def main():
	images = [
			  "zero",
			  "five",
			  "sun",
			  "seven",
			  ]
	n_images = images + ["zero2"]

	#work(2, "pics", images, n_images)
	visualize("pics", images, n_images)
	

if __name__ == "__main__":
	main()