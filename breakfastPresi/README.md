# Bibliometric Analysis Presentation

This directory contains a reveal.js presentation for the Bibliometric Analysis project.

## Getting Started

### Local Development

To view and edit the presentation locally:

1. Start a local server in this directory:

```bash
# If you have Python installed
python -m http.server

# Or with Node.js
npx serve
```

2. Open your browser and navigate to `http://localhost:8000` (for Python) or `http://localhost:3000` (for Node.js)

### File Structure

- `index.html`: Main presentation file
- `dist/`: Core reveal.js files (do not modify)
- `plugin/`: reveal.js plugins (do not modify)
- `css/`: Custom CSS styles (add your own here)
- `assets/`: Images, data files, and other media for your presentation
  - `assets/images/`: Store presentation images here
  - `assets/data/`: Store data files here

## Creating Slides

### Basic Structure

Each slide is contained in a `<section>` element:

```html
<section>
    <h2>Slide Title</h2>
    <p>Slide content goes here</p>
</section>
```

### Adding Images

To add images from your analysis:

```html
<section>
    <h2>Analysis Results</h2>
    <img src="assets/images/your-chart.png" alt="Description">
</section>
```

### Code Snippets

To include code snippets:

```html
<section>
    <h2>Example Code</h2>
    <pre><code class="python">
import pandas as pd

# Your Python code here
    </code></pre>
</section>
```

## Advanced Features

### Vertical Slides

Create nested vertical slides:

```html
<section>
    <section>Main slide</section>
    <section>Vertical slide 1</section>
    <section>Vertical slide 2</section>
</section>
```

### Markdown Support

Write slides using Markdown:

```html
<section data-markdown>
    <textarea data-template>
        ## Slide Title
        
        * Bullet point 1
        * Bullet point 2
        
        ---
        
        ## Next Slide
    </textarea>
</section>
```

### Speaker Notes

Add private speaker notes:

```html
<section>
    <h2>Slide with notes</h2>
    <p>Visible content</p>
    <aside class="notes">
        Private notes visible in presenter view.
        Press 'S' to open speaker view.
    </aside>
</section>
```

## Resources

- [reveal.js Documentation](https://revealjs.com/)
- [reveal.js GitHub Repository](https://github.com/hakimel/reveal.js/)

<p align="center">
  <a href="https://revealjs.com">
  <img src="https://hakim-static.s3.amazonaws.com/reveal-js/logo/v1/reveal-black-text-sticker.png" alt="reveal.js" width="500">
  </a>
  <br><br>
  <a href="https://github.com/hakimel/reveal.js/actions"><img src="https://github.com/hakimel/reveal.js/workflows/tests/badge.svg"></a>
  <a href="https://slides.com/"><img src="https://static.slid.es/images/slides-github-banner-320x40.png?1" alt="Slides" width="160" height="20"></a>
</p>

reveal.js is an open source HTML presentation framework. It enables anyone with a web browser to create beautiful presentations for free. Check out the live demo at [revealjs.com](https://revealjs.com/).

The framework comes with a powerful feature set including [nested slides](https://revealjs.com/vertical-slides/), [Markdown support](https://revealjs.com/markdown/), [Auto-Animate](https://revealjs.com/auto-animate/), [PDF export](https://revealjs.com/pdf-export/), [speaker notes](https://revealjs.com/speaker-view/), [LaTeX typesetting](https://revealjs.com/math/), [syntax highlighted code](https://revealjs.com/code/) and an [extensive API](https://revealjs.com/api/).

---

Want to create reveal.js presentation in a graphical editor? Try <https://slides.com>. It's made by the same people behind reveal.js.

---

### Getting started
- 🚀 [Install reveal.js](https://revealjs.com/installation)
- 👀 [View the demo presentation](https://revealjs.com/demo)
- 📖 [Read the documentation](https://revealjs.com/markup/)
- 🖌 [Try the visual editor for reveal.js at Slides.com](https://slides.com/)
- 🎬 [Watch the reveal.js video course (paid)](https://revealjs.com/course)

--- 
<div align="center">
  MIT licensed | Copyright © 2011-2024 Hakim El Hattab, https://hakim.se
</div>
