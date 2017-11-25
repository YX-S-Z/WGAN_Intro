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

During the training of discriminator networks $$D$$, we want the discriminator to accept real data and reject generated data, thus for real samples $$x \sim \mathbb{P}_r$$ and generated samples $$G(Z),z \sim \mathbb{P}(z)$$, we want $$D(x)$$ to be large and $$D(G(z))$$ to be small. Thus, the objective function $$V(G, D)$$ suits this cases well: by increasing $$D(x)$$ and decreasing $$D(G(z))$$, we are actually increasing the objective function. On the contrary, while training generator networks $$G$$, we want our discriminator $$D$$ makes mistakes thus decreasing the objective function. 

[This site](https://sigmoidal.io/beginners-review-of-gan-architectures/) will give you more information about the network architecture of GANs.

### Training Algorithm and Theoretical Results:
The training algorithm for GANs from the [GANs Paper](https://arxiv.org/pdf/1406.2661.pdf) is show below:
![Training of GANs](https://github.com/simonzhai/WGAN_Intro/blob/master/pictures/GAN_Training_Algorithm.png)


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
