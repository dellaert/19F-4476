---
# NOT USED! Testing new schedule with the csv data file format; but got the illegal quoting bug due to the built-in csv parser by Jekyll and Ruby
---

# Schedule
<table>
	<thead>
		<tr>
			{% for header in site.data.ccssvv.keys %}
				<td>{{header}}</td>
			{% endfor %}
		</tr>
	</thead>
	<tbody>
		{% for row in site.data.ccssvv.content %}
		<tr>
			{% for col in row %}
				<td>{{col}}</td>
			{% endfor %}
		</tr>
		{% endfor %}
	</tbody>
</table>