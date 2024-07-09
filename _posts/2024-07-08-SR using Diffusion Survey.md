---
layout: single        # 문서 형식
title: Diffusion Models, Image Super-Resolution And Everthing : A Survey         # 제목
categories: Generative Model    # 카테고리
toc: true             # 글 목차
author_profiel: false # 홈페이지 프로필이 다른 페이지에도 뜨는지 여부
sidebar:              # 페이지 왼쪽에 카테고리 지정
    nav: "docs"       # sidebar의 주소 지정
#search: false # 블로그 내 검색 비활성화
---
# Keywords
Super Resolution, Diffusion Models, 


# 1. Introduction
Computer-Vision에서 Image Super-Resolution(SR)은 어설픈 특성때문에 오랫동안 어려운 분야로 취급받아왔다. 다양한 분야에서 여러 어려움들을 해결하고자 했고 GAN을 이용하고자 했지만 GAN에서의 regularization과  optimization으로는 극복하기 어려웠다. 최근에 Diffusion Models(DMs)의 등장으로 GAN의 여러 한계들을 뛰어넘게 되었고, SR 역시 Diffusion을 이용해 발전을 이루었다. 하지만 DMs 역시 여러 문제와 한계점이 존재하고 이를 뛰어넘기 위해 많은 연구자들이 힘쓰고 있다.


# 2. Image Super-Resolution
목표 : 저해상도 이미지를 고해상도로 변환하는 것을 목표로 한다.
분야 : 변환시키는 이미지의 수에 따라 Single과 Multi로 구분한다. 

## - 1. Single Image SR(SISR)
주어진 단일 저해상도 이미지 $\mathbf{x} \in \mathbb{R}^{\bar{w} \times \bar{h} \times c}$ 에 대응하는 $\mathbf{y} \in \mathbb{R}^{w \times h \times c}$ 를 생성하는데, 여기서
$\bar{w} < w$, $ \bar{h} < h $ 를 만족한다. 
$x$와 $y$의 관계를 표현하면 다음과 같다.
$\begin{center}$
$\mathbf{x} = D(\mathbf{y};\Theta) = ((\mathbf{y} \otimes \mathbf{k}) \downarrow_{s}$ + $\mathbf{n})_{JPEG_q}$
여기서 $D$는 degradation mapping으로 $D : \mathbb{R}^{w \times h \times c} \rightarrow \mathbb{R}^{\bar{w} \times \bar{h} \times c} $ 이고 $\Theta$는 blur $\mathbf{k}$, noise $\mathbf{n}$, scaling $s$, compression quality $q$ 등과 같은 degradation parameter들을 포함한다. 
화질저하(degradation)는 보통 알 수 없으므로 $D$의 매개변수 $\theta$ 의 inverse mapping을 결정하는 것이 주요 과제이다. 이는 보통 SR 모델로 구현된다. 그리고 원래 HR 이미지인 $\mathbf{y}$와 예측한 SR 이미지인 $\hat{\mathbf{y}}$ 간 차이를 최소화 하는 것을 목표로 하고 이를 다음과 같이 표현할 수 있다.
$\theta = argmin_{\theta} \mathcal{L}(\hat{\mathbf{y}}, \mathbf{y}) + \lambda \phi(\theta)$
여기서 $mathcal{L}$은 원래(실제) HR 이미지인 $\mathbf{y}$와 예측한 SR 이미지인 $\hat{\mathbf{y}}$ 간 손실함수, $\lambda$는 balancing parameter, $\phi(\theta)$는 regularization term이다.



## - 2. Mulit-Image SR(MISR) 
$$



# 3. Diffusion Models 
# 4. Improvements for Diffusion Models
# 5. Diffusion-based Zero-shot SR
# 6. Domain-specific applications
# 7. Discussion and Future work




