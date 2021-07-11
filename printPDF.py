from Code2pdf.code2pdf import *
ifile,ofile,size = "triplet_arch_all.py", "triplet_arch_all.pdf", "A4"
pdf = Code2pdf(ifile, ofile, size)  # create the Code2pdf object
pdf.init_print()    # call print method to print pdf
