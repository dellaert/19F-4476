---
title: Schedule
---

<html>
  <head>
    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="chrome=1">
    <title>{{ page.title }}</title>

    <link rel="stylesheet" href="stylesheets/styles.css">
    <link rel="stylesheet" href="stylesheets/github-light.css">
    <meta name="viewport" content="width=device-width">
    <!--[if lt IE 9]>
    <script src="//html5shiv.googlecode.com/svn/trunk/html5.js"></script>
    <![endif]-->
  </head>
  <body>
    <div class="wrapper">
      <!--header-->
        <h1>Introduction to Computer Vision</h1>
        <p>Website for Fall 2019 edition of CS 4476 at Georgia Tech</p>

        <!--p class="view"><a href="https://github.gatech.edu/gt-cv-class/19F-4476">View the Project on GitHub <small>gt-cv-class/19F-4476</small></a></p-->


      <!--/header-->
      <!--section-->
      <!--table border="1"-->
      <table>
		<thead>
			<tr>
				<td><strong>Date</strong></td>
				<td><strong>Topic</strong></td>
				<td><strong>Slide</strong></td>
				<td><strong>Reading</strong></td>
				<td><strong>Projects</strong></td>
			</tr>
		</thead>
		<tbody>
			{% for item in site.data.schedule %}
			<tr>
				<td>{{item.date}}</td>
				<td>{{item.topic}}</td>
				<td>{{item.slide}}</td>
				<td>{{item.reading}}</td>
				<td>{{item.project}}</td>
			</tr>
			{% endfor %}
		</tbody>
	</table>
        
      <!--/section-->
      <!--footer>
        <p>This project is maintained by <a href="https://github.gatech.edu/gt-cv-class">gt-cv-class</a></p>
        <p><small>Hosted on GitHub Pages &mdash; Theme by <a href="https://github.com/orderedlist">orderedlist</a></small></p>
      </footer-->
    </div>
    <script src="javascripts/scale.fix.js"></script>
    
  </body>
</html>

