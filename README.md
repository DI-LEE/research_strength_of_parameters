# Research about strength of parameters for accuracy in CNN
Research in System Software Research Laboratory https://sites.google.com/view/limseungho

### 배경

딥러닝 모델을 실제 임베디드 시스템에 이식하는 경우, 모델의 크기와 속도가 중요한 문제가 된다. 이를 해결하기 위해 가장 간단한 방법은 모델의 파라미터를 줄이는 것이다. 이에 따라, 가중치 값의 비트를 줄이는 연구를 수행했다.

### 연구 내용

ImageNet 데이터셋으로 pretrained 된 CNN 모델들의 weight 파일을 마지막 비트를 0으로 바꾸는 작업을 했다. 이를 통해 ImageNet dataset의 validation dataset 을 통해 accuracy 를 검증했다. 더 많은 마지막 비트를 0으로 바꿈에 따라 정확도가 점점 떨어지는 것을 확인했지만, 마지막 20비트까지 0으로 바꿔도 정확도가 크게 바뀌지 않았다. 이를 통해 weight 파일의 값을 12비트까지만 사용해도 정확도에 큰 영향을 미치지 않아, 메모리 절약에 도움이 될 것으로 보인다.

### 연구 방법

먼저, ImageNet 데이터셋으로 pretrained 된 CNN 모델들의 weight 파일을 로드하고, bit_change 함수를 사용하여 마지막 비트를 0으로 바꿔주었다. 그 후, 이를 사용하여 ImageNet dataset의 validation dataset을 통해 accuracy 를 검증했다.

#### bit_change

```c
//returns n bit number
float bit_change(float x, int n){
    int tmp = *reinterpret_cast<int*>(&x);
    int k = 32 - n;
    tmp >>= k;
    tmp <<= k;
    return *reinterpret_cast<float*>(&tmp);
}
```

 입력받은 `x`의 비트 중에서 `n`개만 남기고 나머지를 0으로 초기화한 `float` 값을 반환하는 함수입니다.

`*reinterpret_cast<int*>(&x)`는 입력받은 `float` 값을 `int`로 형변환하여 `tmp`에 저장합니다. `int` 형태의 `tmp`에 대해서는 비트 이동 연산을 수행하여 `n`비트 이후의 모든 비트를 0으로 초기화합니다. 마지막으로, `*reinterpret_cast<float*>(&tmp)`를 사용하여 `int` 형태로 변환된 `tmp`를 `float`로 다시 변환하여 반환합니다. (IEEE 754 부동 소수점 표준에서만 작동합니다.)

