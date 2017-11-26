<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
# A Brief Introduction to Wasserstein GANs

This blog is written to intuitively introduce the mathematical background of the well known paper [Wasserstein GANs(WGANs)](https://arxiv.org/pdf/1701.07875.pdf). In 2014, a new framework for generative models: [Generative Adversarial Nets(GANs)](https://arxiv.org/pdf/1406.2661.pdf) was introduced using the nowadays deep learning frameworks and achieved great success. However, unlike some other supervised classification tasks, GANs are often found to be **unstable**(the training losses do not converge) and suffers from **mode collapsing**(the generative fail to generative diverse samples). Notice that previous GANs suffer these problems, WGANs, a new GANs framework came out to solve them. 

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
$$C(G)=-\log4 + 2JSD(\mathbb{P}_r||\mathbb{P}_g)$$. By the quality of JS-divergence, we know that: $$JSD(\mathbb{P}||\mathbb{Q})\in[0,\log2]$$. So ideally, when the objective function $$C(G)$$ reaches its minimum, we have $$JSD(\mathbb{P}_r||\mathbb{P}_g)=0$$, which indicates that $$P_r(x)=P_g(x)$$ almost everywhere. 

- __The -log Alternative

In real case, during the training of generator $$G$$, people found out that the cost of generator does not decrease after using SGD, the [GAN tutorial(section 3.2)](https://arxiv.org/pdf/1701.00160.pdf) claims this problem is caused by a saturated cost function of generator. Thus, the tutorial uses another cost function of $$-log(x)$$ instead of $$log(1-x)$$, that is, instead of minimizing $$$C(G)=\underset{x \sim P_g}{\mathbb{E}}[\log \frac{P_g(x)}{P_r(x)+P_g(x)}]$

It seems that by this minmax training process, we will have a generated distribution $$\mathbb{P}_g$$ that is equal to our real distribution $$\mathbb{P}_r$$ almost everywhere, so by playing this minmax game until equilibria, our goal of generating 'authentic' data is achieved. Sadly, this problem is still far from closed.

### Problems in Traditional GANs:
During the training of traditional GANs, we will frequently encounter these two problems: __unstability__ and __mode collapsing__

- __Unstability__

During the training, we frequently found that our generator loss and its variance are increasing, even when their generated samples are getting better(source: [WGANs paper Figure 8](https://arxiv.org/pdf/1701.07875.pdf)):
![image](https://github.com/simonzhai/WGAN_Intro/blob/master/images/Unstable_generator.png?raw=true)

- __Mode Collapsing__

Mode collapsing means that our generator fails to generate various data samples, instead, it 'collapses' into some fix samples(source [WGANs paper Figure 14](https://arxiv.org/pdf/1701.07875.pdf)):
![image](https://github.com/simonzhai/WGAN_Intro/blob/master/images/Mode_collapse.png?raw=true)

From the picture above, although we randomly choose 64 $$z$$ from our prior, many generated results collapse into few images.

## Towards Principled Methods for Training Generative Adversarial Networks
### Introduction
Since the original GANs suffer from mode unstability and mode collapsing, this paper provides rigious proof to say why previous GANs will eventually encouter such issues and a suitable solutions to solve them. 

###

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your 

ages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/simonzhai/simonzhai.github.io/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
