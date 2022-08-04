import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

matplotlib.rcParams['axes.linewidth'] = 3

def image_to_matrix(image_file, threshold=False):
    image_src = cv2.imread(image_file)
    grayImage = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)
    blackAndWhiteImage = grayImage
    if threshold is True:
    	(thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 0, 255, cv2.THRESH_BINARY)
    blackAndWhiteImage = np.where(blackAndWhiteImage == 0, -1, blackAndWhiteImage)
    blackAndWhiteImage = np.where(blackAndWhiteImage == 255, 1, blackAndWhiteImage)
    return blackAndWhiteImage

class HammingNN:
	def __init__(self, act_func=lambda x: x if x > 0 else 0):
		self.M = None
		self.T = None
		self.m = None
		self.eps = None
		self.state = None
		self.prev_state = None
		self.act_func = act_func

	def train(self, *data_samples):
		if len(data_samples) == 0:
			return
		self.m = len(data_samples)
		n = data_samples[0].size
		if self.M is None:
			self.M = np.zeros((n*n, self.m))
			self.T = np.full((self.m, 1), n / 5)
			self.eps = 1 / (2 * n)
		for k in range(self.m):
			d = data_samples[k].flatten()
			for i in range(n):
				self.M[i, k] = d[i] / 2

	def first_layer(self, data):
		data = data.flatten()
		if self.state is None:
			self.state = np.zeros((self.m, 1))
		for j in range(self.m):
			for i in range(data.size):
				self.state[j] += self.M[i, j] * data[i]
			self.state[j] += self.T[j]

	def second_layer(self):
		for j in range(self.m):
			s = 0
			for k in range(self.m):
				if k != j:
					s += self.state[k]
			self.state[j] -= self.eps * s
			self.state[j] = self.act_func(self.state[j])

	def recall(self, data, max_iterations=10):
		self.first_layer(data)
		self.second_layer()
		self.prev_state = self.state
		for iterations in range(max_iterations):
			self.second_layer()
			if (np.array_equal(self.state, self.prev_state)):
				break
			self.prev_state = self.state
		else:
			print("state still changes after")
			print("{} iterations".format(max_iterations))
		return self.state

def main():
	names = ("one", "two", "three")
	imgs = [image_to_matrix("pics/{}.png".format(x)) for x in names]
	#print(imgs)
	nnames = ("two - Copy", "two - Copy2", "two - Copy3")

	for nname in nnames:
		h = HammingNN()
		h.train(*imgs)
		res = h.recall(image_to_matrix("pics/{}.png".format(nname)))
		idx = np.argmax(res)
		print(res)
		#print(names[idx])
	
		fig, axes = plt.subplots(1, len(names) + 1, figsize=(10, 5))
		for i in range(len(names)):
			img = cv2.imread("pics/{}.png".format(names[i]))
			axes[i].imshow(img)
			axes[i].set_title(names[i])
			axes[i].set_xticks([])
			axes[i].set_yticks([])
		color = "lime"
		axes[idx].spines["top"].set_color(color)
		axes[idx].spines["left"].set_color(color)
		axes[idx].spines["right"].set_color(color)
		axes[idx].spines["bottom"].set_color(color)
	
		img = cv2.imread("pics/{}.png".format(nname))
		axes[-1].imshow(img)
		axes[-1].set_title(nname)
		axes[-1].set_xticks([])
		axes[-1].set_yticks([])
		plt.show()


if __name__ == "__main__":
	main()
