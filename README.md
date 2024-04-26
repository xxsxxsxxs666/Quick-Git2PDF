
![image](logo.png)

<h2>⚈ About This</h2>
You can directly run /dist/repo2pdfAPP/repo2pdfAPP.exe to convert repository into PDF. You can use teh repository url or downloaded repository path to convert it into PDF. The GUI is as follows:

# user interface
![image](GUI.png)

Or you can run the Repositary2PDF.py using CLI. The command is as follows:

```bash
python Repositary2PDF.py \
    --dir="C:/Users/username/Desktop/Quick-Git2PDF" \
    --output_dir="C:/Users/username/Desktop/Quick-Git2PDF" \
    ...
```

Here is default configuration options, youcan modify the config.json file to change the configuration.
```json
{
  "wkhtmltopdf": "C:/Program Files/wkhtmltopdf/bin/wkhtmltopdf.exe",
  "overwrite": true,
  "num_multiprocess": -1,
  "git_download_path": null,
  "output_name": null,
  "output_dir": null,
  "style": "colorful",
  "icon_path": "logo_4.png",
  "only_read": [],
  "gitignore_flexible_config":
  [
    "index.html",
    "tools",
    "utils",
    "tests",
    "test"
  ]
}
```


