from queue import Empty, Queue
from threading import Thread
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import simease_ridge as ridge


class RidgeApp:
    def __init__(self, root):
        self.root = root
        self.events = Queue()
        self.conditions = None
        self.result = None
        self.busy = False
        self.closed = False
        self.poll_id = None
        self.shot = tk.StringVar(root)
        self.status = tk.StringVar(root, value='Loading available SMAJ400A tests...')
        self.details = tk.StringVar(root)
        self.prompt_rmse = tk.StringVar(root, value='—')
        self.full_rmse = tk.StringVar(root, value='—')
        self.training_count = tk.StringVar(root, value='—')
        self._build()
        root.protocol('WM_DELETE_WINDOW', self.close)
        self.poll_id = root.after(75, self._poll)
        self.load_tests()

    def _build(self):
        root = self.root
        root.title('SMAJ Waveform Model · Simease Ridge')
        width = min(1220, root.winfo_screenwidth()-80)
        height = min(860, root.winfo_screenheight()-120)
        left = max(0, (root.winfo_screenwidth()-width)//2)
        top = max(20, (root.winfo_screenheight()-height)//2-20)
        root.geometry(f'{width}x{height}+{left}+{top}')
        root.minsize(980, 660)
        root.configure(background='#edf2f7')
        style = ttk.Style(root)
        style.theme_use('clam')
        style.configure('TButton', font=('Segoe UI', 10), padding=(12, 9))
        style.configure('Accent.TButton', background='#2463a6', foreground='white')
        style.map('Accent.TButton', background=[('disabled', '#a6b4c5'), ('active', '#174b85')])
        style.configure('TCombobox', padding=7, font=('Segoe UI', 11))

        header = tk.Frame(root, background='#152e4d', padx=24, pady=16)
        header.pack(fill='x')
        tk.Label(header, text='SMAJ waveform model', font=('Segoe UI', 22, 'bold'),
                 background='#152e4d', foreground='white').pack(anchor='w')
        tk.Label(header, text='Choose a test and compare its prediction with the measured signal.',
                 font=('Segoe UI', 10), background='#152e4d', foreground='#c9daed').pack(anchor='w', pady=(3, 0))

        content = tk.Frame(root, background='#edf2f7', padx=16, pady=16)
        content.pack(fill='both', expand=True)
        sidebar = tk.Frame(content, background='white', width=260, padx=18, pady=20)
        sidebar.pack(side='left', fill='y', padx=(0, 14))
        sidebar.pack_propagate(False)
        tk.Label(sidebar, text='WHAT TEST DO YOU WANT TO MODEL?', font=('Segoe UI', 9, 'bold'),
                 background='white', foreground='#45607e', wraplength=220, justify='left').pack(anchor='w')
        self.selector = ttk.Combobox(sidebar, textvariable=self.shot, state='disabled', width=18)
        self.selector.pack(fill='x', pady=(12, 8))
        self.selector.bind('<<ComboboxSelected>>', self._selection_changed)
        self.count_label = tk.Label(sidebar, text='SMAJ400A tests only', font=('Segoe UI', 9),
                                    background='white', foreground='#62758c')
        self.count_label.pack(anchor='w')
        self.reload_button = ttk.Button(sidebar, text='Reload metadata', command=self.load_tests)
        self.reload_button.pack(fill='x', pady=(10, 0))
        tk.Frame(sidebar, height=1, background='#e1e7ee').pack(fill='x', pady=20)
        tk.Label(sidebar, textvariable=self.details, font=('Segoe UI', 10), justify='left',
                 background='white', foreground='#233c57', wraplength=220).pack(anchor='w')
        self.run_button = ttk.Button(sidebar, text='Model selected test', style='Accent.TButton', command=self.run_selected)
        self.run_button.pack(fill='x', pady=(24, 12))
        tk.Label(sidebar, text='Each prediction is trained on the other SMAJ400A tests. '
                              'The selected test is excluded from training.',
                 font=('Segoe UI', 10), background='white', foreground='#45607e',
                 wraplength=220, justify='left').pack(anchor='w')
        self.export_button = ttk.Button(sidebar, text='Save waveform CSV', command=self.save_csv, state='disabled')
        self.export_button.pack(side='bottom', fill='x', pady=(12, 0))
        tk.Label(sidebar, text='Use the plot toolbar to zoom, pan, or save an image.',
                 font=('Segoe UI', 9), background='white', foreground='#62758c',
                 wraplength=220, justify='left').pack(side='bottom', anchor='w')

        output = tk.Frame(content, background='#edf2f7')
        output.pack(side='left', fill='both', expand=True)
        cards = tk.Frame(output, background='#edf2f7')
        cards.pack(fill='x', pady=(0, 12))
        for title, value in [('Prompt RMSE', self.prompt_rmse), ('Full waveform RMSE', self.full_rmse),
                             ('Training tests', self.training_count)]:
            card = tk.Frame(cards, background='white', padx=15, pady=10)
            card.pack(side='left', fill='x', expand=True, padx=(0, 6))
            tk.Label(card, text=title, background='white', foreground='#62758c', font=('Segoe UI', 9)).pack(anchor='w')
            tk.Label(card, textvariable=value, background='white', foreground='#152e4d',
                     font=('Segoe UI', 18, 'bold')).pack(anchor='w')
        plot = tk.Frame(output, background='white')
        plot.pack(fill='both', expand=True)
        self.figure = Figure(figsize=(9, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot)
        self.toolbar = NavigationToolbar2Tk(self.canvas, plot, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side='bottom', fill='x')
        self.canvas.get_tk_widget().pack(side='top', fill='both', expand=True)
        self._clear_result('Select a test, then click “Model selected test”.')

        footer = tk.Frame(root, background='#e1e9f2', padx=18, pady=8)
        # Reserve the status strip before the expanding plot claims its space.
        footer.pack(side='bottom', fill='x', before=content)
        self.progress = ttk.Progressbar(footer, mode='indeterminate', length=140)
        self.progress.pack(side='right', padx=(12, 0))
        tk.Label(footer, textvariable=self.status, font=('Segoe UI', 9), background='#e1e9f2',
                 foreground='#233c57', anchor='w').pack(side='left', fill='x', expand=True)

    def _set_busy(self, busy):
        self.busy = busy
        self.selector.configure(state='disabled' if busy or self.conditions is None else 'readonly')
        self.run_button.configure(state='disabled' if busy else 'normal')
        self.reload_button.configure(state='disabled' if busy else 'normal')
        self.export_button.configure(state='normal' if self.result is not None and not busy else 'disabled')
        if busy:
            self.progress.start(12)
        else:
            self.progress.stop()

    def _start_worker(self, operation, event):
        self._set_busy(True)

        def work():
            try:
                self.events.put((event, operation()))
            except Exception as error:
                self.events.put(('error', str(error)))

        Thread(target=work, daemon=True, name='smaj-model-worker').start()

    def load_tests(self):
        if self.busy:
            return
        self.conditions = None
        self._clear_result('Reading simease_ridge_metadata.csv...')
        self.status.set('Loading tests from simease_ridge_metadata.csv...')
        self._start_worker(ridge.load_smaj_conditions, 'loaded')

    def _selection_changed(self, event=None):
        if self.conditions is None or not self.shot.get():
            return
        self._update_details()
        self._clear_result(f'Ready to model test {self.shot.get()}.')
        self.status.set('Ready. Choose a test and click Model selected test.')

    def _update_details(self):
        row = self.conditions.loc[self.conditions.shot_id.eq(int(self.shot.get()))].iloc[0]
        load = '1 MΩ' if row.load_ohm == 1e6 else f'{row.load_ohm:g} Ω'
        self.details.set(f'Test {int(row.shot_id)}\n\n'
                         f'Dose rate     {row.dose_rate:.3g} rad/s\n\n'
                         f'Bias              {row.bias_v:g} V\n\n'
                         f'Load             {load}\n\n'
                         f'Pulse width  {row.pcd_fwhm_ns:g} ns')

    def _clear_result(self, text):
        self.result = None
        for value in (self.prompt_rmse, self.full_rmse, self.training_count):
            value.set('—')
        self.export_button.configure(state='disabled')
        self.figure.clear()
        axis = self.figure.add_subplot()
        axis.set_axis_off()
        axis.text(.5, .5, text, ha='center', va='center', fontsize=13, color='#62758c', transform=axis.transAxes)
        self.toolbar.update()
        self.canvas.draw_idle()

    def run_selected(self):
        if self.busy:
            return
        if self.conditions is None:
            self.load_tests()
            return
        shot_id = int(self.shot.get())
        self._clear_result(f'Modeling test {shot_id}...')
        self.status.set(f'Preparing test {shot_id}...')
        self._start_worker(lambda: ridge.model_shot(
            shot_id, progress=lambda text: self.events.put(('progress', text))), 'result')

    def _poll(self):
        if self.closed:
            return
        try:
            while True:
                event, payload = self.events.get_nowait()
                if event == 'progress':
                    self.status.set(payload)
                elif event == 'loaded':
                    self.conditions = payload
                    choices = [str(shot) for shot in payload.shot_id]
                    self.selector.configure(values=choices)
                    if self.shot.get() not in choices:
                        self.shot.set(choices[0])
                    self.count_label.configure(text=f'{len(payload)} tests · SMAJ400A only')
                    self.run_button.configure(text='Model selected test')
                    self._set_busy(False)
                    self._selection_changed()
                elif event == 'result':
                    self.result = payload
                    self.conditions = payload['conditions']
                    self.selector.configure(values=[str(shot) for shot in self.conditions.shot_id])
                    self.count_label.configure(text=f'{len(self.conditions)} tests · SMAJ400A only')
                    self._update_details()
                    ridge.draw_waveform(self.figure, payload['wave'], payload['predicted'],
                                        payload['shot_id'], payload['diode_type'], payload['score'], payload['full'])
                    self.toolbar.update()
                    self.canvas.draw_idle()
                    self.prompt_rmse.set(f"{payload['score']['rmse_v']:.3f} V")
                    self.full_rmse.set(f"{payload['full']['rmse_v']:.3f} V")
                    self.training_count.set(str(len(payload['train_shot_ids'])))
                    self._set_busy(False)
                    self.status.set(f"Test {payload['shot_id']} complete. Prediction trained on the other "
                                    f"{len(payload['train_shot_ids'])} SMAJ400A tests.")
                elif event == 'error':
                    self._set_busy(False)
                    if self.conditions is None:
                        self.run_button.configure(text='Reload tests')
                    self.status.set('Unable to finish. Check the data files and try again.')
                    self._clear_result('No result available. Check the data files and try again.')
                    messagebox.showerror('Unable to model test', payload, parent=self.root)
        except Empty:
            pass
        if not self.closed:
            self.poll_id = self.root.after(75, self._poll)

    def save_csv(self):
        if self.result is None or self.busy:
            return
        path = filedialog.asksaveasfilename(parent=self.root, title='Save predicted waveform',
                                          defaultextension='.csv', filetypes=[('CSV files', '*.csv')],
                                          initialfile=f"{self.result['shot_id']}_ridge_waveform.csv")
        if path:
            try:
                self.result['wave'].assign(predicted_v=self.result['predicted']).to_csv(path, index=False)
                self.status.set(f"Saved waveform for test {self.result['shot_id']}.")
            except OSError as error:
                messagebox.showerror('Unable to save waveform', str(error), parent=self.root)

    def close(self):
        self.closed = True
        if self.poll_id is not None:
            self.root.after_cancel(self.poll_id)
        self.root.destroy()


def main():
    root = tk.Tk()
    RidgeApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()
