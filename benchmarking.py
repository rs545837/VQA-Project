# Create a robust evaluation benchmark for open-ended VLM QA tasks
# In order to evaluate the VLMs, create a curated benchmark for open-ended QA tasks. The benchmark can contain tests from popular benchmarks such as:
# VQAv2
# OVAD
# OK-VQA
# COCO
# Also, consider test cases from the novel approach presented by Ishmam et al using augmented images to test the robustness of VLMs.

from datasets import load_dataset
from PIL import Image
import numpy as np
import cv2
import skimage as sk
from skimage.filters import gaussian
from scipy.ndimage import map_coordinates
from scipy.ndimage import zoom as scizoom
from imageio import imread 
import json 
from typing import List, Dict
import os
from tqdm import tqdm


class MATHVQA:
    def __init__(self) -> None:
        self.dataset = load_dataset("MathLLMs/MathVision", split="test")
        self.__filter_dataset()
        self.__drop_unnecessary_columns()
    
    def __filter_dataset(self):
        # Filter out the open-ended questions
        self.dataset = self.dataset.filter(lambda x: x["options"] == [])
    
    def __drop_unnecessary_columns(self):
        self.dataset = self.dataset.remove_columns(["options", "image", "solution", "level", "subject"])

    def __sample_dataset(self, num_samples: int):
        self.dataset = self.dataset.shuffle(seed=42).select(range(num_samples))
        data = []
        for sample in tqdm(self.dataset):
            data.append({"question": sample["question"], "answer": sample["answer"], "image": np.array(sample["decoded_image"]).tolist()})
        return data
    
    def output_sample(self, num_samples: int, output_path: str):
        data = self.__sample_dataset(num_samples)
        with open(output_path, "w") as f:
            json.dump({"dataset": data}, f)
    
class ImageAugmentation:
    def __init__(self, image: Image, factor: int) -> None:
        self.image = image
        self.factor = factor

    def disk(self, radius, alias_blur=0.1, dtype=np.float32):
        if radius <= 8:
            L = np.arange(-8, 8 + 1)
            ksize = (3, 3)
        else:
            L = np.arange(-radius, radius + 1)
            ksize = (5, 5)
        X, Y = np.meshgrid(L, L)
        aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
        aliased_disk /= np.sum(aliased_disk)

        # supersample disk to antialias
        return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)
    
    def __shot_noise(self):
        c = [.08, .2, 0.5, 0.8, 1.2][self.factor - 1]
        x = np.array(self.image) / 255.
        new_img= np.clip(x+(np.random.poisson( size=x.shape, lam=c)), 0, 1) * 255
        new_img=np.float32(new_img)
        return cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)

    def __gaussian_noise(self):
        c = [.08, .12, 0.18, 0.26, 0.38][self.factor - 1]
        x = np.array(self.image) / 255.
        new_img= np.clip(x+(np.random.normal(size=x.shape, scale=c)), 0, 1) * 255
        new_img=np.float32(new_img)
        return (cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))

    def __impluse_noise(self):
        c = [.03, .06, .09, 0.17, 0.27][self.factor - 1]
        x = sk.util.random_noise(np.array(self.image) / 255., mode='s&p', amount=c)
        return (cv2.cvtColor(np.float32(np.clip(x, 0, 1) * 255), cv2.COLOR_BGR2RGB))
    
    def __speckle_noise(self):
        c = [.15, .2, 0.35, 0.45, 0.6][self.factor - 1]
        x = np.array(self.image) / 255.
        return (cv2.cvtColor(np.float32(np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255), cv2.COLOR_BGR2RGB))

    def addNoise(self):
        noise_types = [self.__shot_noise, self.__gaussian_noise, self.__impluse_noise, self.__speckle_noise]
        noise_type = np.random.choice(noise_types)
        return noise_type()

    def __defocus_blur(self):
        c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][self.factor - 1]
        x = np.array(self.image) / 255.
        kernel = self.disk(radius=c[0], alias_blur=c[1])

        channels = []
        for d in range(3):
            channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
        channels = np.array(channels).transpose((1, 2, 0))  # 3x224x224 -> 224x224x3
        return (cv2.cvtColor(np.float32(np.clip(channels, 0, 1) * 255), cv2.COLOR_BGR2RGB))
    
    def __glass_blur(self):
        c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][self.factor - 1]
        x = np.uint8(gaussian(np.array(self.image) / 255., sigma=c[0], channel_axis=-1) * 255)

        # locally shuffle pixels
        for i in range(c[2]):
            for h in range(x.shape[0] - c[1], c[1], -1):
                for w in range(x.shape[1] - c[1], c[1], -1):
                    dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                    h_prime, w_prime = h + dy, w + dx
                    # swap
                    x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]

        return (cv2.cvtColor(np.float32(np.clip(gaussian(x / 255., sigma=c[0], channel_axis=-1), 0, 1) * 255), cv2.COLOR_BGR2RGB))
    
    def __clipped_zoom(self, img, zoom_factor):
        h = img.shape[0]
        # ceil crop height(= crop width)
        ch = int(np.ceil(h / float(zoom_factor)))

        w= img.shape[1]
        ch2 = int(np.ceil(w / float(zoom_factor)))
        top = (h - ch) // 2
        side= (w - ch2) // 2
        img = scizoom(img[top:top + ch, side:side + ch2], (zoom_factor, zoom_factor, 1), order=1)
        # trim off any extra pixels
        trim_top = (img.shape[0] - h) // 2
        trim_side = (img.shape[1] - w) // 2


        return img[trim_top:trim_top + h, trim_side:trim_side + w]
    
    def __zoom_blur(self):
        c = [np.arange(1, 1.11, 0.01),
             np.arange(1, 1.16, 0.01),
             np.arange(1, 1.21, 0.02),
             np.arange(1, 1.26, 0.02),
             np.arange(1, 1.31, 0.03)][self.factor - 1]
        x = (np.array(self.image) / 255.).astype(np.float32)
        out = np.zeros_like(x)
        for zoom_factor in c:
            temp = self.__clipped_zoom(x, zoom_factor)
            out += temp


        x = (x + out) / (len(c) + 1)
        return (cv2.cvtColor(np.float32(np.clip(x, 0, 1) * 255), cv2.COLOR_BGR2RGB))

    def addBlur(self):
        blur_types = [self.__defocus_blur, self.__glass_blur, self.__zoom_blur]
        blur_type = np.random.choice(blur_types)
        return blur_type()
    
    def __brightness_transform(self):
        c = [.1, .2, .3, .4, .5][self.factor - 1]
        x = np.array(self.image) / 255.
        x = sk.color.rgb2hsv(x)
        x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
        x = sk.color.hsv2rgb(x)
        return (cv2.cvtColor(np.float32(np.clip(x, 0, 1) * 255), cv2.COLOR_BGR2RGB))
    
    def __contrast_transform(self):
        c = [0.4, .3, .2, .1, .05][self.factor - 1]
        x = np.array(self.image) / 255.
        means = np.mean(x, axis=(0, 1), keepdims=True)
        return (cv2.cvtColor(np.float32(np.clip((x - means) * c + means, 0, 1) * 255), cv2.COLOR_BGR2RGB))
    
    def __saturation_transform(self):
        c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][self.factor - 1]
        x = np.array(self.image) / 255.
        x = sk.color.rgb2hsv(x)  
        x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
        x = sk.color.hsv2rgb(x)
        return (cv2.cvtColor(np.float32(np.clip(x, 0, 1) * 255), cv2.COLOR_BGR2RGB))
    
    def addAttributeTransform(self):
        attribute_types = [self.__brightness_transform, self.__contrast_transform, self.__saturation_transform]
        attribute_type = np.random.choice(attribute_types)
        return attribute_type()
    
    def __elastic_transform(self):
        c = [(244 * 2, 244 * 0.7, 244 * 0.1),   # 244 should have been 224, but ultimately nothing is incorrect
             (244 * 2, 244 * 0.08, 244 * 0.2),
             (244 * 0.05, 244 * 0.01, 244 * 0.02),
             (244 * 0.07, 244 * 0.01, 244 * 0.02),
             (244 * 0.12, 244 * 0.01, 244 * 0.02)][self.factor - 1]

        image = np.array(self.image, dtype=np.float32) / 255.
        shape = image.shape
        shape_size = shape[:2]

        # random affine
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32([center_square + square_size,
                           [center_square[0] + square_size, center_square[1] - square_size],
                           center_square - square_size])
        pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

        dx = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                       c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
        dy = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                       c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
        dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]

        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
        return (cv2.cvtColor(np.float32(np.clip(map_coordinates(image, indices, order=1, mode='reflect').reshape(shape), 0, 1) * 255), cv2.COLOR_BGR2RGB))

    def __spatter_transform(self):
        c = [(0.65, 0.3, 4, 0.69, 0.6, 0),
             (0.65, 0.3, 3, 0.68, 0.6, 0),
             (0.65, 0.3, 2, 0.68, 0.5, 0),
             (0.65, 0.3, 1, 0.65, 1.5, 1),
             (0.67, 0.4, 1, 0.65, 1.5, 1)][self.factor - 1]
        x = np.array(self.image, dtype=np.float32) / 255.

        liquid_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])

        liquid_layer = gaussian(liquid_layer, sigma=c[2])
        liquid_layer[liquid_layer < c[3]] = 0
        if c[5] == 0:
            liquid_layer = (liquid_layer * 255).astype(np.uint8)
            dist = 255 - cv2.Canny(liquid_layer, 50, 150)
            dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
            _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
            dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
            dist = cv2.equalizeHist(dist)
            ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
            dist = cv2.filter2D(dist, cv2.CV_8U, ker)
            dist = cv2.blur(dist, (3, 3)).astype(np.float32)

            m = cv2.cvtColor(liquid_layer * dist, cv2.COLOR_GRAY2BGRA)
            m /= np.max(m, axis=(0, 1))
            m *= c[4]

            # water is pale turqouise
            color = np.concatenate((175 / 255. * np.ones_like(m[..., :1]),
                                    238 / 255. * np.ones_like(m[..., :1]),
                                    238 / 255. * np.ones_like(m[..., :1])), axis=2)

            color = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
            x = cv2.cvtColor(x, cv2.COLOR_BGR2BGRA)

            return (cv2.cvtColor(np.clip(x + m * color, 0, 1), cv2.COLOR_BGRA2RGB) * 255)
        else:
            m = np.where(liquid_layer > c[3], 1, 0)
            m = gaussian(m.astype(np.float32), sigma=c[4])
            m[m < 0.8] = 0

            # mud brown
            color = np.concatenate((63 / 255. * np.ones_like(x[..., :1]),
                                    42 / 255. * np.ones_like(x[..., :1]),
                                    20 / 255. * np.ones_like(x[..., :1])), axis=2)

            color *= m[..., np.newaxis]
            x *= (1 - m[..., np.newaxis])

            return (cv2.cvtColor(np.float32(np.clip(x + color, 0, 1) * 255), cv2.COLOR_BGR2RGB))

    def addPhysical(self):
        physical_types = [self.__elastic_transform, self.__spatter_transform]
        physical_type = np.random.choice(physical_types)
        return physical_type()

    def __pixelate_transform(self):
        c = [0.6, 0.5, 0.4, 0.3, 0.15][self.factor - 1]
        x = np.array(self.image)
        # x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        width = int(x.shape[1] * c)
        height = int(x.shape[0] * c)
        dim = (width, height)
        resized = cv2.resize(x, dim, interpolation = cv2.INTER_AREA)
        return (cv2.cvtColor(np.float32(resized), cv2.COLOR_BGR2RGB))
    
    def __jpeg_compression_transform(self):
        c = [25, 18, 15, 10, 7][self.factor - 1]
        x = np.array(self.image)
        cv2.imwrite("temp.jpg", x, [int(cv2.IMWRITE_JPEG_QUALITY), c]) 
        temp = imread("temp.jpg")
        return temp

    def addDigital(self):
        digital_types = [self.__pixelate_transform, self.__jpeg_compression_transform]
        digital_type = np.random.choice(digital_types)
        return digital_type()
    
def get_augments(image, factor):
    augments = [
        np.array(ImageAugmentation(image, factor).addNoise()).tolist(),
        np.array(ImageAugmentation(image, factor).addBlur()).tolist(),
        np.array(ImageAugmentation(image, factor).addAttributeTransform()).tolist(),
        np.array(ImageAugmentation(image, factor).addPhysical()).tolist(),
        np.array(ImageAugmentation(image, factor).addDigital()).tolist()
    ]
    return augments

class VQAV2:
    def __init__(self, question_path: str, answer_path: str, image_directory: str) -> None:
        with open(question_path, "r") as f:
            self.questions = json.load(f)["questions"]
        with open(answer_path, "r") as f:
            self.answers = json.load(f)["annotations"]
        self.image_directory = image_directory
    
    def __extract_unique_answers(self, answers: List[Dict]) -> List[str]:
        unique_answers = set()
        for answer in answers:
            unique_answers.add(answer["answer"])
        return list(unique_answers)
    
    def __sample_dataset(self, num_samples: int) -> List[Dict]:
        # Sample num_samples random indices from the dataset
        indices = np.random.choice(range(len(self.questions)), num_samples, replace=False)
        data = []
        for idx in tqdm(indices):
            question = self.questions[idx]["question"]
            answer = self.__extract_unique_answers(self.answers[idx]["answers"])
            image_id = self.questions[idx]["image_id"]
            # Left Zero-pad the image_id to 12 digits
            image_id = f"COCO_val2014_{image_id:012d}"
            image_path = os.path.join(self.image_directory, f"{image_id}.jpg")
            image = Image.open(image_path)
            data.append({"question": question, "answer": answer, "image": np.array(image).tolist()})
        return data
    
    def output_sample(self, num_samples: int, output_path: str):
        data = self.__sample_dataset(num_samples)
        with open(output_path, "w") as f:
            json.dump({"dataset": data}, f)

class OKVQA:
    def __init__(self, question_path: str, answer_path: str, image_directory: str) -> None:
        with open(question_path, "r") as f:
            self.questions = json.load(f)["questions"]
        with open(answer_path, "r") as f:
            self.answers = json.load(f)["annotations"]
        self.image_directory = image_directory
    
    def __extract_unique_answers(self, answers: List[Dict]) -> List[str]:
        unique_answers = set()
        for answer in answers:
            unique_answers.add(answer["answer"])
        return list(unique_answers)
    
    def __sample_dataset(self, num_samples: int) -> List[Dict]:
        # Sample num_samples random indices from the dataset
        indices = np.random.choice(range(len(self.questions)), num_samples, replace=False)
        data = []
        for idx in tqdm(indices):
            question = self.questions[idx]["question"]
            answer = self.__extract_unique_answers(self.answers[idx]["answers"])
            image_id = self.questions[idx]["image_id"]
            # Left Zero-pad the image_id to 12 digits
            image_id = f"COCO_val2014_{image_id:012d}"
            image_path = os.path.join(self.image_directory, f"{image_id}.jpg")
            image = Image.open(image_path)
            data.append({"question": question, "answer": answer, "image": np.array(image).tolist()})
        return data
    
    def output_sample(self, num_samples: int, output_path: str):
        data = self.__sample_dataset(num_samples)
        with open(output_path, "w") as f:
            json.dump({"dataset": data}, f)

class A_VQA:
    def __init__(self, question_path: str, answer_path: str, image_directory: str) -> None:
        with open(question_path, "r") as f:
            self.questions = json.load(f)["questions"]
        with open(answer_path, "r") as f:
            self.answers = json.load(f)["annotations"]
        self.image_directory = image_directory
    
    def __extract_unique_answers(self, answers: List[Dict]) -> List[str]:
        unique_answers = set()
        for answer in answers:
            unique_answers.add(answer["answer"])
        return list(unique_answers)
    
    def __sample_dataset(self, num_samples: int, factor: int) -> List[Dict]:
        # Sample num_samples random indices from the dataset
        indices = np.random.choice(range(len(self.questions)), num_samples, replace=False)
        data = []
        for idx in tqdm(indices):
            question = self.questions[idx]["question"]
            answer = self.__extract_unique_answers(self.answers[idx]["answers"])
            image_id = self.questions[idx]["image_id"]
            # Left Zero-pad the image_id to 12 digits
            image_id = f"COCO_val2014_{image_id:012d}"
            image_path = os.path.join(self.image_directory, f"{image_id}.jpg")
            image = Image.open(image_path)
            
            # Get 5 augmented images
            augments = get_augments(image, factor)
            for augment in augments:
                data.append({"question": question, "answer": answer, "image": augment})
        return data
    
    def output_sample(self, num_samples: int, output_path: str, factor: int):
        # Divide the number of samples to account for the fact that there are 5 questions per image using augments
        num_samples = num_samples // 5
        data = self.__sample_dataset(num_samples, factor)
        with open(output_path, "w") as f:
            json.dump({"dataset": data}, f)

def create_benchmarks(num_samples: int):
    # Create benchmarks for VQAv2, MATHVQA, OKVQA, and A-VQA

    # VQAv2
    vqa = VQAV2(
        question_path="./VQA/VQA-questions.json",
        answer_path="./VQA/VQA-annotations.json",
        image_directory="./MSCOCO/val2014"
    )
    vqa.output_sample(num_samples, f"VQAv2-{num_samples}.json")

    # MATHVQA
    mathvqa = MATHVQA()
    mathvqa.output_sample(num_samples, f"MATHVQA-{num_samples}.json")

    # OKVQA
    okvqa = OKVQA(
        question_path="./OKVQA/OKVQA-questions.json",
        answer_path="./OKVQA/OKVQA-annotations.json",
        image_directory="./MSCOCO/val2014"
    )
    okvqa.output_sample(num_samples, f"OKVQA-{num_samples}.json")

    # A-VQA
    avqa = A_VQA(
        question_path="./VQA/VQA-questions.json",
        answer_path="./VQA/VQA-annotations.json",
        image_directory="./MSCOCO/val2014"
    )

    for factor in range(1, 6):
        avqa.output_sample(num_samples, f"A-VQA-{num_samples}-{factor}.json", factor)

