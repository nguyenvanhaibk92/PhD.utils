## All you need to present your work (by Hai Nguyen in Pho Group - UT austin)

## Why you need this?

- Beamer in LaTeX is the best to a lot of things, but it is not easy to deal with videos, animations, etc. and rearrangements.
- PowerPoint is the best to deal with videos, animations, etc. but it is not easy to deal with math expressions, etc.

- Can we instantly update the latex slides in PowerPoint? Yes, we can! Then, you can take the best of both worlds.

## Description
This is the folder to show you all how to organize files to use the best features of Beamer in LaTeX and PowerPoint

- nice math expressions
- nice template
- nice animations
- nice transitions
- instant global formatting
- ... all the best 

## You need to have:

- LaTeX installed
- PowerPoint installed
- IguanaTex installed in PowerPoint, https://github.com/Jonathan-LeRoux/IguanaTex

## What are included in this folder?

- Latex:

  - Latex/main.tex: the main file to compile to get a beamer set of slides
  - Latex/slides_<N>.tex (N=1,2,3,...): the files to include in main.tex (any update there can be synchronized to PowerPoint)
  - Latex/Figs: the folder to store all figures
  - Latex/template: the folder to store Dr. Bui's favorite template/macros/etc. (you can change to your own)

- PowerPoint:

  - based_line.pptx: the PowerPoint file that is synchronized with main.tex

## How to use?

- On the Latex:

  - **Step 0:** change this path to your local computer path in the main.tex file
  ```
  \def\@rootpath{/Users/hainguyenvan/Desktop/Github_Clone/PhD.utils/Presentation_template/Latex/}
  ```
  to
  ```
  \def\@rootpath{<YOUR_DIRECTORY_PATH>/PhD.utils/Presentation_template/Latex/}
  ```
  
  - Step 1: Prepare your slides in Latex in slides_<N>.tex files
  - Step 2: Change template_1.tex to your favorite template and information
  - Step 3: Compile main.tex to get the beamer slides

- On the PowerPoint:

  - Step 1: Open based_line.pptx
  - Step 2: Go to IguanaTex tab, choose "Edit Latex Display", and update the path as **Step 0** in the Latex.
  - Step 3: Done! You can now synchronize your slides in PowerPoint with the beamer slides in Latex.
  - Step 4:  You can change the bottom-page information about presenters, talk tile, Group, etc. by going to View -> Slide Master
