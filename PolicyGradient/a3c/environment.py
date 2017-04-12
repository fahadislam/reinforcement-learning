from PIL import Image
import json
import os
import numpy as np

class Environment:
	def __init__(self, target_object = 'nature_valley_sweet_and_salty_nut_almond', \
								root = '../../../ActiveVisionDataset/', \
								scene_name = 'Home_001_1', \
								images_dir = 'jpg_rgb', \
								env_first_image = '000110000010101.jpg', \
								instances_id_path = '../../../ActiveVisionDataset/'):
		self.scene_name = scene_name
		self.env_first_image = env_first_image

		## load annotations
		with open(os.path.join(root,scene_name,'annotations.json')) as json_data:
			self.data = json.loads(json_data.read())
			self.current_image = self.env_first_image

		## load instances_id map
		with open(instances_id_path + 'instance_id_map.txt') as instance_id:
			for line in instance_id:
				line = line.replace('\n','')
				instance_id = line.split(" ")
				if instance_id[0] == target_object:
					self.target_object_id = int(instance_id[1])
					# print (self.target_object_id)

		self.load_images(root,scene_name,images_dir,self.data.keys())

	def load_images(self,root,scene_name,images_dir,image_names):
		self.images = {}
		for image_name in image_names:
			img = (Image.open(os.path.join(root,scene_name,images_dir,image_name)))
			self.images[image_name] = self.process_state_for_memory(img)

	def process_state_for_memory(self,im):
		im.load()
		im = Image.fromarray(np.uint8(im))
		im_gray = im.convert('L')
		return np.uint8(np.asarray(im_gray))

	def process_state_for_network(self,im):
		return np.float32(im)/255.0

	def reset(self): 		
		self.current_image = self.env_first_image
		return self.process_state_for_network(self.images[self.current_image])

	def step(self,action):
		done = False
		reward = 0
		next_image = self.data[self.current_image][action]
		if next_image == "":
			done = True
		for bbox in self.data[self.current_image]["bounding_boxes"]:
			if bbox[4] == self.target_object_id:
				reward = 1
				done = True
				break
		self.current_image = next_image
		return self.process_state_for_network(self.images[self.current_image]),reward,done,"" ## no info

if __name__ == '__main__':
	env = Environment()
	first_image = env.reset()
	# result = Image.fromarray((first_image * 255).astype(np.uint8))
	# result.save(env.current_image)
	done = False
	while(not done):
		new_image,reward,done,info = env.step("rotate_ccw")
		print(reward,done,info)
		# result = Image.fromarray((new_image * 255).astype(np.uint8))
		# result.save(env.current_image)
