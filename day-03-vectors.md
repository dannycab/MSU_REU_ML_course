---
marp: true
theme: king
paginate: true

title: Day 03 - Vectors
description: Slides for PHY 321 Spring 2025, Day 03: Vectors
author: Prof. Danny Caballero <caball14@msu.edu>
keywords: classical mechanics, vectors, differential equations, motion
url: https://dannycaballero.info/phy321msu/slides/day-03-vectors.html
---

# Day 03 - Vectors

$$\mathbf{r} = r_x\hat{x} + r_y\hat{y} + r_z\hat{z}$$
$$\mathbf{r} = r_x\hat{i} + r_y\hat{j} + r_z\hat{k}$$
$$\mathbf{r} = r_x\hat{e}_1 + r_y\hat{e}_2 + r_z\hat{e}_3$$
$$\mathbf{r} = \langle r_x, r_y, r_z\rangle$$

![bg right:40%](./images/vector.jpeg)

---

# Announcements

* Homework 1 is due next Friday
* Help sessions will start next week
* Last class was not recorded properly
* No class on Monday (MLK Day)
* Complete the [student information poll](https://forms.office.com/r/bqFghtWrbj) by Monday, please

---

# Goals for this week

## Be able to answer the following questions.

* What is Classical Mechanics?
* How can we formulate it?
* What are the essential physics models for single particles?
* What mathematics do we need to get started?

---

# Reminders from Day 02

* Classical Physics has existed for centuries
* Newton's Laws are one formulation of Classical Mechanics
* We can use Newton's Laws to describe the motion of particles
* Newton's Laws give rise to differential equations

--- 

# Example: Ball Falling in 1D in Air

We derived the following differential equation for the motion of a ball falling in air:

$$m\ddot{y} = mg - b v - c v^2$$

We argued for low speeds, we neglect the $v^2$ term. 

$$m\ddot{y} = mg - b v$$

We can instead write this differential equation for $v$:

$$\dot{v} = g - \frac{b}{m}v$$

---

# Example: Ball Falling in 1D in Air

Is this integrable? **Yes!**

$$\frac{dv}{dt} = g - \frac{b}{m}v$$

$$\frac{dv}{g - \frac{b}{m}v} = dt$$

$$\int \frac{dv}{g - \frac{b}{m}v} = \int dt$$

We will come back to this next week.

---

<!-- # Example: Ball Falling in 1D in Air

Let's perform a $u$-substitution:

Let $u = g - \frac{b}{m}v$, then $du = -\frac{b}{m}dv$

$$\int \frac{du}{u} = -\frac{m}{b}\int dt$$

$$\ln|u| = -\frac{m}{b}t + C$$

$$u = e^{-\frac{m}{b}t + C} = A e^{-\frac{m}{b}t}$$

where $A = e^C$, a constant.

---

# Example: Ball Falling in 1D in Air

In this limit, we are able to find an analytical solution for the velocity of the ball:

$$u = A e^{-\frac{m}{b}t}$$

$$v = \frac{m}{b}g - A e^{-\frac{m}{b}t}$$

If the ball starts at rest, $v(0) = 0$, then $A = \frac{m}{b}g$.

$$v = \frac{m}{b}g(1 - e^{-\frac{m}{b}t})$$

In the limit of large times, $t \rightarrow \infty$, $v \rightarrow \frac{m}{b}g$, the terminal velocity.

--- -->

# Vector Properties

Newtonian Mechanics is a vector theory. Here are a few mathematical properties of vectors:

* **Addition**: $\mathbf{A} + \mathbf{B} = (A_x + B_x)\hat{x} + (A_y + B_y)\hat{y} + (A_z + B_z)\hat{z}$
* **Scalar Multiplication**: $c\mathbf{A} = \langle cA_x, cA_y, cA_z\rangle$
* **Dot Product**: $\mathbf{A}\cdot\mathbf{B} = A_xB_x + A_yB_y + A_zB_z = AB\cos\theta$
* **Cross Product**: $\mathbf{A}\times\mathbf{B} = \langle A_yB_z - A_zB_y, A_zB_x - A_xB_z, A_xB_y - A_yB_x\rangle$
* **Unit Vectors**: $\hat{A} = \frac{\mathbf{A}}{|\mathbf{A}|} \qquad |\hat{A}| = 1$

---

# Generative AI

*Generative AI* is a type of artificial intelligence that can generate new data from existing data.

* It is an **extractive** technology that has mined a vast data set.
* It is a **probabilistic** technology that uses statistical models to generate new data.
* It is **not** a **creative** technology that can generate new ideas, concepts, or products.
* It is **not** a **truthful** technology that can generate new data that is intrinsically true.

**The "Grow At Any Cost" approach to generative AI is destroying communities, violating federal and international laws, upending climate progress, and consolidating power in the hands of a few.** 

---

# Generative AI Energy Consumption

![bg right:20%](./images/a100.png)

* The max power of a single A100 chip is 400W.
* The compute needed to perform a simple generation task takes roughly 1s.
* One request thus uses about 400J.
* If everyone in class performs one request, we would use roughly 40,000 J.
    * That will charge your cell phone from 0-100% about 5 times.
* If everyone on campus performs one request, we would use roughly 30,000,000 J.
    * That will charge your cell phone from 0-100% everyday for 11 years.


---

# And yet,

* Generative AI can be used productively.
* Generative AI can support accessibility.
* Generative AI can support creativity.
* Generative AI can support learning.

The complexity and tension of these issues are why we need to develop a policy together.

**I will not live with the consequences of Generative AI, but y'all will, so this policy must be yours.**

---


# Creating a Generative Artificial Intelligence Policy

We define **productivity** as the ability to use Generative AI to deepen your understanding of Classical Mechanics.

## Take five minutes to answer the following for yourself:

1. What are ways that you think that AI can be used productively in our classroom?
2. What are ways that you think that AI can be used unproductively in our classroom?
3. What do you think are acceptable uses of AI in our classroom?
4. What do you think are unacceptable uses of AI in our classroom?
5. How should we document the use of AI in our classroom?
6. Once we define a policy, how should we collectively enforce it?

---

# Creating a Generative Artificial Intelligence Policy

We define **productivity** as the ability to use Generative AI to deepen your understanding of Classical Mechanics.

## Share your ideas are your table. Develop a consensus on the following:

1. What do y'all think are acceptable uses of AI in our classroom?
2. What do y'all think are unacceptable uses of AI in our classroom?
3. How should we document the use of AI in our classroom?
4. Once we define a policy, how should we collectively enforce it?

Add your answers to the form at the following link: [https://forms.office.com/r/Bsh6ugKQ9Y](https://forms.office.com/r/Bsh6ugKQ9Y)