# Preparing the dataset with CVAT

These steps download the Kaggle dataset, review its boxes in a local CVAT and export them for `scripts/prepare_dataset.py`. Commands run from the project folder unless stated otherwise.

## 1. Download the dataset

Create an API token in Kaggle under **Settings > API** and save `kaggle.json` to `~/.kaggle/` (on Windows, `C:\Users\<you>\.kaggle\`). Never commit this file.

```bash
pip install -r requirements-dev.txt
kaggle datasets download -d andrewmvd/helmet-detection -p data/raw --unzip
```

This creates `data/raw/images/` and `data/raw/annotations/`, each with 764 files.

## 2. Package the files for CVAT

```bash
python scripts/make_cvat_zips.py
```

This creates `images.zip` and `voc_annotations.zip` in `data/cvat_import/`. The script builds the `Annotations/` folder layout that CVAT's Pascal VOC importer expects.

## 3. Start CVAT

Clone CVAT outside this project. Docker Desktop needs to be running.

```bash
git clone --depth 1 https://github.com/cvat-ai/cvat --config core.autocrlf=input
cd cvat
docker compose up -d
docker exec -it cvat_server bash -ic "python3 ~/manage.py createsuperuser"
```

- `--depth 1` downloads only the latest commit, as the full history is large enough to fail on a slow connection.
- `core.autocrlf=input` keeps Windows from changing line endings in scripts that run in Linux containers.
- If `createsuperuser` warns about unapplied migrations, CVAT is still starting. Wait a minute and run it again.

Open <http://localhost:8080> and log in.

## 4. Create the project and task

1. Create a project with two rectangle labels, spelled exactly like the XML files: `With Helmet` and `Without Helmet`.
2. Inside the project, create a task and upload `images.zip`.
3. In the task, choose **Actions > Upload annotations**, select **Pascal VOC 1.1** and upload `voc_annotations.zip`.

## 5. Review the boxes

Open the job and page through the images: `F` goes to the next frame, `D` to the previous one and `N` starts a new box. Save with `Ctrl+S`.

## 6. Export

Choose **Projects > your project > Actions > Export dataset**, then select **Ultralytics YOLO Detection 1.0**. The images are not needed in the export, as they are taken from `data/raw/images/`. Unzip the result into `data/cvat_export/`:

```powershell
Expand-Archive path\to\export.zip data\cvat_export
```

## 7. Build the dataset

```bash
python scripts/prepare_dataset.py
```

This validates the labels, leaves out images with corrupt coordinates, and writes the split and `data.yaml` to `data/processed/`.

## Stop CVAT

From the CVAT folder:

```bash
docker compose down
```

Your CVAT data stays in Docker volumes. Adding `-v` to this command would delete it.
