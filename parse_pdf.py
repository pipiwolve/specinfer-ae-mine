import sys
try:
    import Quartz.PDFKit as PDFKit
    from Foundation import NSURL
    pdfURL = NSURL.fileURLWithPath_("todolist.pdf")
    pdfDoc = PDFKit.PDFDocument.alloc().initWithURL_(pdfURL)
    if pdfDoc:
        print(pdfDoc.string())
except ImportError:
    print("PyObjC not installed")
