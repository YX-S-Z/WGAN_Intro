<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
# A Brief Introduction to Wasserstein GANs

This blog is written to __intuitively__ introduce the mathematical background of the well known paper [Wasserstein GANs(WGANs)](https://arxiv.org/pdf/1701.07875.pdf), for detailed proof, please refer to the original papers. In 2014, a new framework for generative models: [Generative Adversarial Nets(GANs)](https://arxiv.org/pdf/1406.2661.pdf) was introduced using the nowadays deep learning frameworks and achieved great success. However, unlike some other supervised classification tasks, GANs are often found to be **difficult**(the generator generate nothing but garbage), **unstable**(the training losses do not converge), and suffers from **mode collapsing**(the generative fail to generative diverse samples). Notice that previous GANs suffer these problems, WGANs, a new GANs framework came out to solve them. 

This blog will introduce 3 papers:

- [Generative Adversarial Nets(GANs)](https://arxiv.org/pdf/1406.2661.pdf)

- [Towards Principled Methods for Training Generative Adversarial Networks](https://arxiv.org/pdf/1701.04862.pdf)

- [Wasserstein GANs](https://arxiv.org/pdf/1701.07875.pdf)


## Generative Adversarial Nets
### Introduction
GANs introduced a new adversarial framework for training generative models: given some real samples(say images), simultaneously train a __generator(G)__ and a __discriminator(D)__, where **D** is trained to classify the real samples from those generative samples while **G** is trained to let **D** make mistakes during classification. 

### Objective Function and its Mathematical Intuition:
The objective function of GANs is this:
$$ V(G, D) = \underset{G}{\min} \underset{D}{\max} \underset{x \sim \mathbb{P}_r}{\mathbb{E}}[\log D(x)] + \underset{z \sim \mathbb{P}}{\mathbb{E}}[\log (1-D(G(z)))] $$
Where the generator $$G(z)$$ is a network that generate a real sample(image) by deconvolution and $$z$$ is an input from random noise distribution(normal distribution or uniform distribution) $$\mathbb{P}(z)$$. The discriminator $$D(x)$$ is a network(function) that represents the probability that our input samples(images) $$x$$ came from the real data rather than generative data, which indicates that: $$D(x) \in [0,1]$$. 

During the training of discriminator networks $$D$$, we want the discriminator to accept real data and reject generated data, thus for real samples $$x \sim \mathbb{P}_r$$ and generated samples $$G(Z),z \sim \mathbb{P}$$, we want $$D(x)$$ to be large and $$D(G(z))$$ to be small. Thus, the objective function $$V(G, D)$$ suits this cases well: by increasing $$D(x)$$ and decreasing $$D(G(z))$$, we are actually increasing the objective function. On the contrary, while training generator networks $$G$$, we want our discriminator $$D$$ makes mistakes thus decreasing the objective function. 

[This site](https://sigmoidal.io/beginners-review-of-gan-architectures/) will give you more information about the network architecture of GANs.

### Training Algorithm and Theoretical Results:
The training algorithm for GANs from the [GANs Paper](https://arxiv.org/pdf/1406.2661.pdf) is show below:
![Image](https://github.com/simonzhai/WGAN_Intro/blob/master/images/GAN_Training_Algorithm.png?raw=true)

- __Analysis of Discriminator__

For given generator $$G$$, we can substitute $$D(G(z))$$ with $$D(x)$$, because for any noise $$z$$ in the prior $$\mathbb{P}$$, $$G(z)$$ will generate a data sample(image). So the optimal discriminator $$D$$ will maximize our objective function(assuming the density functions are continuous):

$$V(G,D)=\int_xP_r(x)\log(D(x))+P_g(x)\log(1-D(x))dx$$ 

And by solving the gradient with respect to $$D(x)$$, we know that the optimal discriminator is $$D^*_G(x)=\frac{P_r(x)}{P_r(x)+P_g(x)}$$.

- __Analysis of Generator__

According to the training algorithm, we start training our generator when our discriminator is welled trained, ideally, our discriminator $$D^*_G(x)=\frac{P_r(x)}{P_r(x)+P_g(x)}$$. So during the training of generator, we want to minimize our objective function

$$C(G)=\underset{x \sim \mathbb{P}_r}{\mathbb{E}}[\log \frac{P_r(x)}{P_r(x)+P_g(x)}] + \underset{x \sim P_g}{\mathbb{E}}[\log \frac{P_g(x)}{P_r(x)+P_g(x)}]$$

Using some trick in [KL divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) and [Jensen–Shannon divergence](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence), our objective function for $$G$$ can be written in this way: 
$$C(G)=-\log4 + 2JSD(\mathbb{P}_r||\mathbb{P}_g)$$. By the quality of JS-divergence, we know that: $$JSD(\mathbb{P}||\mathbb{Q})\in[0,\log2]$$. So ideally, when the objective function $$C(G)$$ reaches its minimum, we have $$JSD(\mathbb{P}_r||\mathbb{P}_g)=0$$, which indicates that $$P_r(x)=P_g(x)$$ [almost everywhere](https://en.wikipedia.org/wiki/Almost_everywhere). 

- __The -log Alternative__

In real case, during the training of generator $$G$$, people found out that the cost of generator does not decrease after using SGD, the [GAN tutorial(section 3.2)](https://arxiv.org/pdf/1701.00160.pdf) claims this problem is caused by a saturated cost function of generator. Thus, the tutorial uses another cost function of $$-\log(x)$$ instead of $$\log(1-x)$$, that is, instead of minimizing $$C_1(G)=\underset{z \sim P}{\mathbb{E}}[\log (1-D(G(z)))]$$, we minimize $$C_2(G)=\underset{z \sim P}{\mathbb{E}}[-\log (D(G(z))]$$. The difference between these two functions are shown in the following picture:
![image](https://github.com/simonzhai/WGAN_Intro/blob/master/images/-log_alternative.png?raw=true)
Where the blue curve represents $$y=-\log(x)$$ and the the red curve represents $$y=\log(1-x)$$. Since our generator $$G$$ is updated after our discriminator $$D$$ is well trained, thus our discriminator will be stronger than our generator and $$D(G(z))$$ will be very small, say close to 0. In this case, we can learn from the picture above that the gradient of the red curve is much flatter than the blue curve, which means that SGD will have less effect on the first cost function $$C_1(G)$$. And thus, using $$C_2(G)$$ as an alternative seems to be a wiser choice in this case. 

It seems that by this minmax training process, we will have a generated distribution $$\mathbb{P}_g$$ that is equal to our real distribution $$\mathbb{P}_r$$ almost everywhere, so by playing this minmax game until equilibria, our goal of generating 'authentic' data is achieved. Sadly, this problem is still far from closed.

### Problems in Traditional GANs:
During the training of traditional GANs, we will frequently encounter these three problems: __difficulty__, __instability__, and __mode collapsing__

- __Difficulty__

Not all training of GANs will finally generate meaningful results, what sometimes happens is that while the discriminator gets better during training, generator will fail and eventually generate garbage(source: [WGANs paper Figure 12](https://arxiv.org/pdf/1701.07875.pdf)):
![image](https://github.com/simonzhai/WGAN_Intro/blob/master/images/Generator_faliure.png?raw=true)

- __Instability__

During the training, we frequently found that our generator loss and its variance are increasing, even when their generated samples are getting better(source: [WGANs paper Figure 8](https://arxiv.org/pdf/1701.07875.pdf)):
![image](https://github.com/simonzhai/WGAN_Intro/blob/master/images/Unstable_generator.png?raw=true)

- __Mode Collapsing__

Mode collapsing means that our generator fails to generate various data samples, instead, it 'collapses' into some fix samples(source [WGANs paper Figure 14](https://arxiv.org/pdf/1701.07875.pdf)):
![image](https://github.com/simonzhai/WGAN_Intro/blob/master/images/Mode_collapse.png?raw=true)

From the picture above, although we randomly choose 64 $$z$$ from our prior, many generated results collapse into few images.

## Towards Principled Methods for Training Generative Adversarial Networks
### Introduction
Since the original GANs suffers from mode __unstability__ and __mode collapsing__, this paper provides rigious proof to say why previous GANs will eventually encouter those two issues and provides a __better cost function(or a better metric to evaluate the 'similarity' between two probability distributions)__ to avoid these issues. 

### The reasons for failure in training GANs

In real cases, we can prove that there is always a **perfect** discriminator $$D^*(x)$$ that can perfectly distinguish real data from generated data, and gradient descend method **has no effect** on this discriminator $$D^*(x)$$, which explains why our discriminator gets better and our generator fails during training.

- __Perfect Discriminator Theorem([Section 2.1](https://arxiv.org/pdf/1701.04862.pdf))__

Assume that the [supports](https://en.wikipedia.org/wiki/Support_(mathematics)) of our real sample distribution $$\mathbb{P}_r$$ and our generated sample distribution $$\mathbb{P}_g$$ are [submanifolds](https://en.wikipedia.org/wiki/Submanifold) $$\mathcal{M}$$ and $$\mathcal{P}$$ in our feature space $$\mathcal{X}$$(the vector space of final fully connected layer in the discriminator network). Then we can always find a optimal discriminator $$D(x)\rightarrow[0,1]$$, s.t. $$D(x)=1,(x\in\mathcal{M})$$, $$D(x)=0,(x\in\mathcal{P})$$ and $$\nabla_xD(x)=0,({x\in\mathcal{M}\cup\mathcal{P}})$$ [almost everywhere](https://en.wikipedia.org/wiki/Almost_everywhere).

To intuitively understand this theorem, we can divide this problem in two parts: $$\mathcal{M}\cap\mathcal{P}=\emptyset$$ and $$\mathcal{M}\cap\mathcal{P}\neq\emptyset$$ 

(a) When: $$\mathcal{M}\cap\mathcal{P}=\emptyset$$ 

This means that the intersect between the supports of $$\mathbb{P}_r$$ and $$\mathbb{P}_g$$ is empty. In this cases, the following picture will provide some intuitive explanations(for detailed proof, check theorem 2.1 in [this paper](https://arxiv.org/pdf/1701.04862.pdf)):
![image](https://github.com/simonzhai/WGAN_Intro/blob/master/images/Perfect_descriminator_below.png?raw=true)
In the picture above, assume that $$\mathcal{M}$$ is the red submanifold and $$\mathcal{P}$$ is the green submanifold, both of them are 2 dimensional manifolds in 3 dimensional space. An obvious optimal discriminator will be a sigmiod like surface, which classifies all points above the blue manifold with true and below blue manifold with fake. An interesting attribute of sigmoid like function is that it suffers from saturated gradients, in the picture above, we can learn that this discriminator(blue manifold) can perfectly discriminate these two manifolds and gradient descend does not work on the supports of the $$\mathcal{M}$$(green manifold) and $$\mathcal{P}$$(red manifold).

(b) When: $$\mathcal{M}\cap\mathcal{P}\neq\emptyset$$

To understand the proof in this case, we have to introduce a mathematical idea of [Transversal Intersection](http://mathworld.wolfram.com/TransversalIntersection.html) and [perfectly aligned(definition 2.2)](https://arxiv.org/pdf/1701.04862.pdf) between two manifolds. If you don't understand the math behind these two idea, it is totally fine, these following two pictures will give you some idea about perfect aligned manifolds and not perfectly aligned manifolds:
![image](https://github.com/simonzhai/WGAN_Intro/blob/master/images/perfectly_align.png?raw=true)

__two perfectly aligned circles in 3 dimensional space__

![image](https://github.com/simonzhai/WGAN_Intro/blob/master/images/not_perfectly_align.png?raw=true)

__two not perfectly aligned circles in 3 dimensional space__

![image](https://github.com/simonzhai/WGAN_Intro/blob/master/images/Not_perfectly_align_gaussian.png?raw=true)

__two not perfectly aligned gaussian spheres in 3 dimensional space__

From the previous pictures we can have some intuitive concepts about __perfectly aligned__: when two manifolds $$\mathcal{M}$$ and $$\mathcal{P}$$ are not perfectly aligned, the [measure](https://en.wikipedia.org/wiki/Measure_(mathematics)) of the $$\mathcal{M}\cap\mathcal{P}$$ intersect strictly less than the measure of both $$\mathcal{M}$$ and $$\mathcal{P}$$. Also, [lemma 2](https://arxiv.org/pdf/1701.04862.pdf) tells us that in real case, the probability that two random distributions are perfectly aligned are actually extremely small(equals to 0). 
