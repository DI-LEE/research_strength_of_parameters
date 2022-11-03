#!/bin/bash

images_folder=/home/dilee/Desktop/pico-cnn_research/accuracy_test/test_images

##predict_arr=() #예측 결과값


#16비트
#폴더 내 이미지파일들을 읽고 예측하는 for 루프
for image_path in $images_folder/*
do
	#예측할 이미지의 경로
	echo $image_path

	#모델 실행
	result1=`/home/dilee/Desktop/pico-cnn/onnx_import/generated_code/vgg16/vgg16 /home/dilee/Desktop/pico-cnn/onnx_import/generated_code/vgg16/network.weights.bin /home/dilee/Desktop/pico-cnn/onnx_import/generated_code/vgg16/imagenet.means /home/dilee/Desktop/pico-cnn/data/imageNet_labels/LOC_synset_mapping.txt $image_path 32`

	result2=`/home/dilee/Desktop/pico-cnn/onnx_import/generated_code/vgg16/vgg16 /home/dilee/Desktop/pico-cnn/onnx_import/generated_code/vgg16/network.weights.bin /home/dilee/Desktop/pico-cnn/onnx_import/generated_code/vgg16/imagenet.means /home/dilee/Desktop/pico-cnn/data/imageNet_labels/LOC_synset_mapping.txt $image_path 24`

	result3=`/home/dilee/Desktop/pico-cnn/onnx_import/generated_code/vgg16/vgg16 /home/dilee/Desktop/pico-cnn/onnx_import/generated_code/vgg16/network.weights.bin /home/dilee/Desktop/pico-cnn/onnx_import/generated_code/vgg16/imagenet.means /home/dilee/Desktop/pico-cnn/data/imageNet_labels/LOC_synset_mapping.txt $image_path 16`

	result4=`/home/dilee/Desktop/pico-cnn/onnx_import/generated_code/vgg16/vgg16 /home/dilee/Desktop/pico-cnn/onnx_import/generated_code/vgg16/network.weights.bin /home/dilee/Desktop/pico-cnn/onnx_import/generated_code/vgg16/imagenet.means /home/dilee/Desktop/pico-cnn/data/imageNet_labels/LOC_synset_mapping.txt $image_path 12`

	result5=`/home/dilee/Desktop/pico-cnn/onnx_import/generated_code/vgg16/vgg16 /home/dilee/Desktop/pico-cnn/onnx_import/generated_code/vgg16/network.weights.bin /home/dilee/Desktop/pico-cnn/onnx_import/generated_code/vgg16/imagenet.means /home/dilee/Desktop/pico-cnn/data/imageNet_labels/LOC_synset_mapping.txt $image_path 8`


	##predict_arr+=("$result")
	echo $result1 >> predict_label_32bit.txt
	echo $result2 >> predict_label_24bit.txt
	echo $result3 >> predict_label_16bit.txt
	echo $result4 >> predict_label_12bit.txt
	echo $result5 >> predict_label_8bit.txt
done


##
#predict.txt에 저장
#for predict in "${predict_arr[@]}"; do
#		echo $predict >> predicted.txt
#done
