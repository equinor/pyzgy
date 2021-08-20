#!/usr/bin/env python3

import os
import time
from datetime import datetime
import traceback
import tkinter as tk
import tkinter.filedialog
import tkinter.messagebox
import tkinter.ttk as ttk
from PIL import Image, ImageTk

from ..tools import copy

class TemporaryWidget(tk.Frame):
    def __init__(self, parent=None):
        super().__init__(parent, bd=2)
        self._parent = parent
        self.create_widgets()

    def create_widgets(self):
        if False:
            load = Image.open("dialog.png")
            render = ImageTk.PhotoImage(load)
            self.orig_label = tk.Label(self, image=render)
            self.orig_label.image = render
        elif False:
            self.orig_label = tk.Label(self, text="Compress or decompress ZGY files")
        else:
            self.orig_label = tk.Label(self)
        self.orig_label.grid(sticky="nsew")

class RadioOne(tk.Frame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._parent = parent
        self.create_widgets()

    def create_widgets(self):
        tk.Radiobutton(self, text="Lossy", padx=0, justify=tk.LEFT,
                       variable=self._parent._parent.compressmode,
                       value=1).grid(row=0, column=1)
        tk.Radiobutton(self, text="Lossless", padx=0, justify=tk.LEFT,
                       variable=self._parent._parent.compressmode,
                       value=2).grid(row=0, column=2)
        tk.Radiobutton(self, text="Uncompressed", padx=0, justify=tk.LEFT,
                       variable=self._parent._parent.compressmode,
                       value=3).grid(row=0, column=3)

class UserInputWidget(tk.Frame):
    def __init__(self, parent=None):
        super().__init__(parent, relief=tk.GROOVE, bd=2)
        self._parent = parent
        self.create_widgets()

    def create_widgets(self):
        self._lbl_inputfile = tk.Label(self, text="Input file:")
        self._lbl_inputfile.grid(row=0, column=0, sticky="w")
        self._txt_inputfile = tk.Entry(self, width=50, textvariable=self._parent.inputname)
        self._txt_inputfile.grid(row=0, column=1, sticky="ew")
        self._btn_inputfile_select = tk.Button(self, text="...", command=self._parent.open_input_dialog)
        self._btn_inputfile_select.grid(row=0, column=2, sticky="w")

        self._lbl_outputfile = tk.Label(self, text="Output file:")
        self._lbl_outputfile.grid(row=1, column=0, sticky="w")
        self._txt_outputfile = tk.Entry(self, width=50, textvariable=self._parent.outputname)
        self._txt_outputfile.grid(row=1, column=1, sticky="ew")
        self._btn_outputfile_select = tk.Button(self, text="...", command=self._parent.open_output_dialog)
        self._btn_outputfile_select.grid(row=1, column=2, sticky="w")

        self._lbl_radio_one = tk.Label(self, text="Output is:")
        self._lbl_radio_one.grid(row=2, column=0, sticky="w")
        self._radio_one = RadioOne(self)
        self._radio_one.grid(row=2, column=1, columnspan=2, sticky="w")

        self._scale_sqnr = tk.Scale(self, orient=tk.HORIZONTAL, from_=20, to=70, tickinterval=10, label="Signal/noise ratio in dB", resolution=5, variable=self._parent.snr)
        self._scale_sqnr.set(30)
        self._scale_sqnr.grid(row=3, column=0, columnspan=3, sticky="eww")

        self._lbl_1 = tk.Label(self, text="<-- Smaller files    Better fidelity -->")
        self._lbl_1.grid(row=4, column=0, columnspan=3, sticky="eww")

        self.grid_columnconfigure(1, weight=1)

class ProgressWidget(tk.Frame):
    def __init__(self, parent=None):
        super().__init__(parent, bd=2)
        self._parent = parent
        self.create_widgets()

    def create_widgets(self):
        self._lbl_last_completed_hdr = tk.Label(self, text="Last completed")
        self._lbl_input_size_hdr = tk.Label(self, text="Input size")
        self._lbl_output_percent_hdr = tk.Label(self, text="New size")
        self._lbl_output_size_hdr = tk.Label(self, text="New size")

        self._lbl_last_completed = tk.Label(self, text="(name)")
        self._lbl_input_size = tk.Label(self, text="XX MB")
        self._lbl_output_percent = tk.Label(self, text="XX %")
        self._lbl_output_size = tk.Label(self, text="XX MB")

        self._lbl_last_completed_hdr.grid(row=0, column=2, padx=5, sticky="w")
        self._lbl_input_size_hdr.grid(row=0, column=3, padx=5)
        self._lbl_output_percent_hdr.grid(row=0, column=4, padx=5)
        self._lbl_output_size_hdr.grid(row=0, column=5, padx=5)

        self._lbl_last_completed.grid(row=1, column=2, padx=5, sticky="w")
        self._lbl_input_size.grid(row=1, column=3, padx=5)
        self._lbl_output_percent.grid(row=1, column=4, padx=5)
        self._lbl_output_size.grid(row=1, column=5, padx=5)

        self._lbl_time_elapsed = tk.Label(self, text="00:00:00", font='TkFixedFont')
        self._lbl_time_elapsed.grid(row=2, column=0, sticky="w")
        self._lbl_time_remain = tk.Label(self, text="00:00:00", font='TkFixedFont')
        self._lbl_time_remain.grid(row=3, column=0)
        self._lbl_elapsed = tk.Label(self, text="elapsed")
        self._lbl_elapsed.grid(row=2, column=1)
        self._lbl_remain = tk.Label(self, text="remain")
        self._lbl_remain.grid(row=3, column=1)
        self._pb_progress = ttk.Progressbar(self)
        self._pb_progress.grid(row=2, column=2, rowspan=2, columnspan=4, sticky="ew")

        self.grid_columnconfigure(2, weight=1)

class BottomWidget(tk.Frame):
    def __init__(self, parent=None):
        super().__init__(parent, bd=2)
        self._parent = parent
        self.create_widgets()

    def create_widgets(self):
        self._estimate = tk.Button(self, text="Estimate size...", state=tk.DISABLED)
        self._estimate.grid(row=0, column=0, padx=5)
        self._run = tk.Button(self, text="Run", command=self._parent.clicked_run)
        self._run.grid(row=0, column=1, padx=5)
        self._stop = tk.Button(self, text="Stop", state=tk.DISABLED, command=self._parent.clicked_stop)
        self._stop.grid(row=0, column=2, padx=5)
        self._quit = tk.Button(self, text="Quit", command=self._parent.master.destroy)
        self._quit.grid(row=0, column=3, padx=5)

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.grid(sticky="nsew")
        self.create_variables()
        self.create_widgets()
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)
        self._running = False
        self._stopping = False
        self._has_run = False
        self._start_time = 0
        self._stop_time = 0
        self.update_dependencies()

    def report_callback_exception(self, *args):
        err = traceback.format_exception(*args)
        tkinter.messagebox.showerror("Internal error", err)
        self.master.destroy()

    def set_overwrite_confirmed(self, value):
        """
        True means the output file does not exist or that the user has
        answered "Yes" to the prompt about overwriting. Note that the
        file selection dialog pops up this question itself, so if the
        output file was filled in from there it will already be confirmed.
        Otherwise we wait until the user clicks "Run" before we check.
        """
        #print("overwrite_confirmed(" + str(bool(value)) + ")")
        self.confirmed = value

    def validate(self):
        if not self.inputname.get():
            tkinter.messagebox.showerror("Error", "Input file must be specified.")
            return False
        elif not os.path.exists(self.inputname.get()):
            tkinter.messagebox.showerror("Error", "Input file does not exist.")
            return False

        if not self.outputname.get():
            tkinter.messagebox.showerror("Error", "Output file must be specified.")
            return False
        elif self.outputname.get() == self.inputname.get():
            tkinter.messagebox.showerror("Error", "Input and output cannot be the same file.")
            return False
            # Note that user can sneak around this when typing the name,
            # e.g. foo.zgy and ./foo.zgy.

        elif not self.confirmed:
            if not os.path.exists(self.outputname.get()):
                self.confirmed = True
            else:
                self.confirmed = tkinter.messagebox.askokcancel(
                    "Question",
                    'File "{0}" already exists.\nDo you want to overwrite it?'.format(self.outputname.get()),
                    icon=tkinter.messagebox.WARNING)
        if not self.confirmed:
            return False
        return True

    def update_dependencies(self):
        self.set_state(self._user_input_widget, not self._running)
        self.set_state(self._user_input_widget._scale_sqnr,
                       self.compressmode.get() == 1 and not self._running)
        self.set_state(self._user_input_widget._lbl_1,
                       self.compressmode.get() == 1 and not self._running)
        self.set_state(self._progress_widget, self._running or self._has_run)
        self.set_state(self._bottom_widget._run, not self._running)
        self.set_state(self._bottom_widget._stop, self._running)

    @staticmethod
    def nicenumber(n):
        """
        Format a number with units. Output is always 7 chars.
        """
        if not n: return "       "
        if n < 10000: return "{0:4d}   ".format(n)
        if n < 1024*10000: return "{0:4d} KB".format(n // 1024)
        if n < 1024*1024*10000: return "{0:4d} MB".format(n // (1024*1024))
        return "{0:4d} GB".format(n // (1024*1024*1024))

    def show_results(self):
        isize, osize = (None, None)
        try:
            isize = os.path.getsize(self.inputname.get())
        except Exception:
            pass
        if not self._running and self._has_run:
            try:
                osize = os.path.getsize(self.outputname.get())
            except Exception:
                pass
        isize_str = self.nicenumber(isize) if isize else ""
        osize_str = self.nicenumber(osize) if osize else ""
        osize_pct = "{0:3d} %".format(
            100 * osize // isize) if isize and osize else "     "

        pw = self._progress_widget
        pw._lbl_last_completed_hdr.configure(
            text = "Current file" if self._running else "Last completed")
        shortname = self.inputname.get()
        shortname = os.path.basename(shortname)
        if len(shortname) > 23: shortname = "---" + shortname[-20:]
        pw._lbl_last_completed.configure(text=shortname)
        pw._lbl_input_size.configure(text=isize_str)
        pw._lbl_output_percent.configure(text=osize_pct)
        pw._lbl_output_size.configure(text=osize_str)
        elapsed = "  :  :  "
        elapsed_sec = int(self._stop_time - self._start_time)
        if elapsed_sec > 0 and elapsed_sec < 24*60*60:
            elapsed = datetime.utcfromtimestamp(elapsed_sec).strftime('%H:%M:%S')

        pw._lbl_time_elapsed.configure(text=elapsed)
        pw._lbl_time_remain.configure(text="  :  :  ")
        #self._pb_progress

    def set_state(self, w, state):
        if isinstance(w, tk.Frame):
            for child in w.winfo_children():
                self.set_state(child, state)
        elif isinstance(w, ttk.Progressbar):
            pass # has no "state"
        else:
            s = tk.NORMAL if state else tk.DISABLED
            if False:
                print("Change {0} from {1} to {2}".format(str(w), w.cget('state'), s))
            if isinstance(w, tk.Scale):
                w.configure(state=s, fg="black" if state else "gray")
            w.configure(state=s)

    def set_running(self, on):
        if on and not self._running: self._start_time = time.time()
        if not on and self._running: self._stop_time = time.time()
        self._running = on
        self._stopping = False
        if on: self._has_run = True
        self.update_dependencies()
        self.show_results()

    def clicked_run(self):
        #print("RUN", self)
        if self.validate():
            self.set_running(True)
            mode = self.compressmode.get()
            snr = int(self.snr.get()) if mode==1 else 99 if mode==2 else 0
            cmd = [
                "python3", "-m", "openzgy.tools.copy",
                "--force", "float",
                "--snr", str(snr),
                self.inputname.get(),
                self.outputname.get(),
            ]
            print(*cmd)
            self.set_overwrite_confirmed(False)
            if True:
                copy.copy(
                    self.inputname.get(), self.outputname.get(),
                    progress1=self.update_progress,
                    progress2=lambda done, total: self.update_progress(done, total, True),
                    forcetype='float', snr=snr)
                if self._stopping:
                    # User clicked 'Stop', so the output, if it exists,
                    # is most likely corrupt.
                    self.update_progress(0, 100)
                    if tkinter.messagebox.askokcancel(
                            "Question",
                            'You clicked "Stop",\nso the output file\nis probably unusable.\nDelete it now?', icon=tkinter.messagebox.WARNING):
                        try:
                            os.remove(self.outputname.get())
                        except IOError as ex:
                            tkinter.messagebox.showerror("Error", "While deteling output:\n" + str(ex))
                else:
                    self.update_progress(100, 100, True)
                # If the actual copy is disabled,
                # clicked_stop needs to do this for us.
                self.set_running(False)
        else:
            self.set_running(False)

    def update_progress(self, done, total, flushing=False):
        percent = int(50.0 * done / total)
        if done != 0: percent = max(1, percent)
        if done >= total: percent = 50
        if flushing: percent += 50
        self._progress_widget._pb_progress['value'] = percent
        if False:
            print("DONE:", self._progress_widget._pb_progress['value'],
                  "of", self._progress_widget._pb_progress['maximum'],
                  "flushing LODs" if flushing else "copying")
        root.update()
        return self._running and not self._stopping

    def clicked_stop(self):
        #print("STOP", self)
        self._stopping = True
        # Only when debugging, when the copy process isn't started.
        # Otherwise it will be called when the process finishes.
        #self.set_running(False)

    def showvar(self, name):
        var = getattr(self, name)
        value = var.get()
        if value is str: value = '"' + value + '"'
        print('trace {0} {1} = {2}'.format(name, var, value))

    def create_variables(self):
        self.confirmed = False
        self.inputname = tk.StringVar(self)
        self.outputname = tk.StringVar(self)
        self.outputname.trace("w", lambda *args: self.set_overwrite_confirmed(False))
        self.compressmode = tk.IntVar(self, 1)
        self.compressmode.trace("w", lambda *args: self.update_dependencies())
        self.snr = tk.DoubleVar(self, 30)

        if False: # debug
            self.inputname.trace("w", lambda *args: self.showvar("inputname"))
            self.outputname.trace("w", lambda *args: self.showvar("outputname"))
            self.compressmode.trace("w", lambda *args: self.showvar("compressmode"))
            self.snr.trace("w", lambda *args: self.showvar("snr"))

    def create_widgets(self):
        #self._temporary_widget = TemporaryWidget(self)
        #self._temporary_widget.grid(row=0, column=0, sticky="we")

        self._user_input_widget = UserInputWidget(self)
        self._user_input_widget.grid(row=1, column=0, padx=5, pady=5, sticky="ew")

        self._progress_widget = ProgressWidget(self)
        self._progress_widget.grid(row=2, column=0, sticky="eww")

        self._bottom_widget = BottomWidget(self)
        self._bottom_widget.grid(row=3, column=0, sticky="se", padx=5, pady=5)

    def open_input_dialog(self):
        name = tk.filedialog.askopenfilename(filetypes=[("ZGY", "*.zgy")])
        if name:
            self.inputname.set(name)

    def open_output_dialog(self):
        name = tk.filedialog.asksaveasfilename(filetypes=[("ZGY", "*.zgy")], defaultextension=".zgy")
        if name:
            self.outputname.set(name)
            self.set_overwrite_confirmed(True)

root, app = (None, None)

def Main():
    global root, app
    root = tk.Tk()
    root.title("Compress or decompress ZGY files")
    root.grid_columnconfigure(0, weight=1)
    root.grid_rowconfigure(0, weight=1)
    app = Application(master=root)
    root.report_callback_exception = app.report_callback_exception
    app.mainloop()

if __name__ == "__main__":
    Main()

# Copyright 2017-2020, Schlumberger
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
