from pypdf import PdfReader


def load_pdf(filepath: str) -> list[str]:
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        text += t
    with open(f"books/{filepath.split('/')[-1]}.txt", "w") as f:
        f.write(text)

    return text.splitlines()


def load_txt(filepath: str) -> list[str]:
    return open(filepath, "r").read().splitlines()


AVAILABLE_LOADERS = {"pdf": load_pdf, "txt": load_txt}
