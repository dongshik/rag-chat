import io
import PyPDF2
import requests


def is_url_path(string: str):
    return (
        string.startswith("http://")
        or string.startswith("https://")
        or string.startswith("www.")
    )


def read_pdf_from_url(url: str):
    response = requests.get(url)

    if response.status_code == 200:
        pdf = io.BytesIO(response.content)
        pdf_reader = PyPDF2.PdfReader(pdf)

        return pdf_reader
    return None


def read_pdf_from_local(file_path: str):
    with open(file_path, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        return pdf_reader


def get_pdf_content_form_url(url: str):
    pdf_data = []
    pdf_reader = read_pdf_from_url(url)
    if not pdf_reader:
        raise Exception("invalid URL")
    num_pages = len(pdf_reader.pages)

    for page_num in range(num_pages):
        page_obj = pdf_reader.pages[page_num]
        text = page_obj.extract_text()
        pdf_data.append({"page_no": page_num + 1, "content": text, "src": url})
    return pdf_data


def get_pdf_content_from_local(file_path: str):
    pdf_data = []

    with open(file_path, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        num_pages = len(pdf_reader.pages)

        for page_num in range(num_pages):
            page_obj = pdf_reader.pages[page_num]
            text = page_obj.extract_text()
            pdf_data.append(
                {
                    "page_no": page_num + 1,
                    "content": text,
                    "src": file_path.split("/")[-1]
                }
            )
    return pdf_data


def get_pdf_content(path: str):
    if is_url_path(path):
        return get_pdf_content_form_url(path)
    return get_pdf_content_from_local(path)