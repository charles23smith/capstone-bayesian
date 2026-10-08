from queue import Empty, Queue
from threading import Thread
from math import isfinite
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from research import simease_ridge as ridge


class RidgeApp:
    def __init__(self, root):
        self.root = root
        self.events = Queue()
        self.conditions = None
        self.result = None
        self.batch_stats = None
        self.busy = False
        self.closed = False
        self.poll_id = None
        self.shot = tk.StringVar(root)
        self.status = tk.StringVar(root, value='Loading available SMAJ400A tests...')
        self.details = tk.StringVar(root)
        self.prompt_rmse = tk.StringVar(root, value='—')
        self.full_rmse = tk.StringVar(root, value='—')
        self.prompt_r2 = tk.StringVar(root, value='R²: —')
        self.full_r2 = tk.StringVar(root, value='R²: —')
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
        actions = tk.Frame(output, background='#edf2f7')
        actions.pack(fill='x', pady=(0, 10))
        self.run_all_button = ttk.Button(actions, text='Run all tests', command=self.run_all_tests,
                                         state='disabled')
        self.run_all_button.pack(side='left')
        self.batch_export_button = ttk.Button(actions, text='Download CSV', command=self.save_all_stats)
        self.metrics_button = ttk.Button(actions, text='View waveform stats',
                                         command=self.show_pulse_metrics, state='disabled')
        self.metrics_button.pack(side='right')
        cards = tk.Frame(output, background='#edf2f7')
        cards.pack(fill='x', pady=(0, 12))
        for title, value, r2 in [('Prompt RMSE', self.prompt_rmse, self.prompt_r2),
                                 ('Full waveform RMSE', self.full_rmse, self.full_r2),
                                 ('Training tests', self.training_count, None)]:
            card = tk.Frame(cards, background='white', padx=15, pady=10)
            card.pack(side='left', fill='x', expand=True, padx=(0, 6))
            tk.Label(card, text=title, background='white', foreground='#62758c', font=('Segoe UI', 9)).pack(anchor='w')
            tk.Label(card, textvariable=value, background='white', foreground='#152e4d',
                     font=('Segoe UI', 18, 'bold')).pack(anchor='w')
            if r2 is not None:
                tk.Label(card, textvariable=r2, background='white', foreground='#45607e',
                         font=('Segoe UI', 11)).pack(anchor='w', pady=(3, 0))
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
        self.status_label = tk.Label(footer, textvariable=self.status, font=('Segoe UI', 9),
                                     background='#e1e9f2', foreground='#233c57', anchor='w')
        self.status_label.pack(side='left', fill='x', expand=True)

    def _set_busy(self, busy):
        self.busy = busy
        self.selector.configure(state='disabled' if busy or self.conditions is None else 'readonly')
        self.run_button.configure(state='disabled' if busy else 'normal')
        self.reload_button.configure(state='disabled' if busy else 'normal')
        self.export_button.configure(state='normal' if self.result is not None and not busy else 'disabled')
        self.metrics_button.configure(state='normal' if self.result is not None and not busy else 'disabled')
        self.run_all_button.configure(state='normal' if self.conditions is not None and not busy else 'disabled')
        self.batch_export_button.configure(state='disabled' if busy else 'normal')
        if not busy:
            self._set_training(False)

    def _set_training(self, training):
        if training:
            self.progress.pack(side='right', padx=(12, 0), before=self.status_label)
            self.progress.start(12)
        else:
            self.progress.stop()
            self.progress.pack_forget()

    def _start_worker(self, operation, event):
        self._set_busy(True)
        self._set_training(event in ('result', 'batch_result'))

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
        self.batch_stats = None
        self.batch_export_button.pack_forget()
        self._clear_result('Reading siamese_ridge_metadata.csv...')
        self.status.set('Loading tests from siamese_ridge_metadata.csv...')
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
        for value in (self.prompt_r2, self.full_r2):
            value.set('R²: —')
        self.export_button.configure(state='disabled')
        self.metrics_button.configure(state='disabled')
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

    def run_all_tests(self):
        if self.busy or self.conditions is None:
            return
        self.batch_stats = None
        self.batch_export_button.pack_forget()
        self.status.set('Running all tests with each test excluded from its own training...')
        self._start_worker(lambda: ridge.all_test_stats(
            progress=lambda text: self.events.put(('progress', text))), 'batch_result')

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
                    self.prompt_r2.set(f"R²: {ridge.format_r2(payload['score']['r2'])}")
                    self.full_r2.set(f"R²: {ridge.format_r2(payload['full']['r2'])}")
                    self.training_count.set(str(len(payload['train_shot_ids'])))
                    self._set_busy(False)
                    self.status.set(f"Test {payload['shot_id']} complete. Prediction trained on the other "
                                    f"{len(payload['train_shot_ids'])} SMAJ400A tests.")
                elif event == 'batch_result':
                    self.batch_stats = payload
                    self._set_busy(False)
                    self.batch_export_button.pack(side='left', padx=(8, 0), after=self.run_all_button)
                    self.status.set(f'All {len(payload)} tests complete. View the table or download CSV.')
                    self.show_all_stats(payload)
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

    def show_pulse_metrics(self):
        if self.result is None or self.busy:
            return
        try:
            comparison = ridge.pulse_comparison(self.result['wave'], self.result['predicted'])
        except ValueError as error:
            messagebox.showerror('Unable to measure pulse', str(error), parent=self.root)
            return
        window = tk.Toplevel(self.root)
        window.title(f"Test {self.result['shot_id']} · Pulse measurements")
        window.transient(self.root)
        window.geometry('900x380')
        window.minsize(760, 340)
        content = ttk.Frame(window, padding=16)
        content.pack(fill='both', expand=True)
        ttk.Label(content, text=f"Test {self.result['shot_id']}: measured vs. held-out prediction",
                  font=('Segoe UI', 12, 'bold')).pack(anchor='w', pady=(0, 12))
        columns = ('metric', 'unit', 'measured', 'predicted', 'absolute_error')
        table = ttk.Treeview(content, columns=columns, show='headings', height=6)
        for column, title, width in zip(columns,
                ('Measurement', 'Unit', 'Measured', 'Predicted', 'RMSE (one test)'),
                (300, 60, 125, 125, 145)):
            table.heading(column, text=title)
            table.column(column, width=width, minwidth=50,
                         anchor='w' if column == 'metric' else 'e')
        for row in comparison.to_dict('records'):
            table.insert('', 'end', values=(row['metric'], row['unit'], *[
                f'{row[key]:.3f}' if isfinite(row[key]) else 'N/A'
                for key in ('measured', 'predicted', 'absolute_error')]))
        table.pack(fill='both', expand=True)
        ttk.Label(content, text='For one test, each scalar RMSE equals its absolute error.\n'
                  'Peak magnitude and crossings use each waveform’s pre-pulse baseline. '
                  'Time to peak is relative to the PCD reference (0 ns).\n'
                  'Area is baseline-subtracted and signed over −60 to 1000 ns. '
                  'N/A means a crossing or required coverage is unavailable.',
                  wraplength=840, justify='left').pack(anchor='w', pady=(12, 0))
        return window

    def show_all_stats(self, data):
        window = tk.Toplevel(self.root)
        window.title(f'All {len(data)} tests · Waveform stats')
        window.transient(self.root)
        width = min(1180, self.root.winfo_screenwidth()-80)
        height = min(620, self.root.winfo_screenheight()-120)
        window.geometry(f'{width}x{height}')
        content = ttk.Frame(window, padding=16)
        content.pack(fill='both', expand=True)
        header = ttk.Frame(content)
        header.pack(fill='x', pady=(0, 12))
        ttk.Button(header, text='Download CSV',
                   command=lambda: self.save_all_stats(data, window)).pack(side='right')
        ttk.Label(header, text=f'{len(data)} tests: model evaluation',
                  font=('Segoe UI', 12, 'bold')).pack(side='left')
        ttk.Label(content, text='Scroll horizontally for all six measurement RMSEs. '
                  'Each test is predicted using the other tests.\n'
                  'Errors are absolute differences (single-test scalar RMSE). '
                  'N/A means unavailable crossings or coverage. CSV preserves full precision.\n'
                  'Summary: waveform scores are per-test means; measurement RMSEs use √mean(error²) across valid tests.',
                  wraplength=width-60, justify='left').pack(anchor='w', pady=(0, 12))
        frame = ttk.Frame(content)
        frame.pack(fill='both', expand=True)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        columns = list(data.columns)
        table = ttk.Treeview(frame, columns=columns, show='headings')
        horizontal = ttk.Scrollbar(frame, orient='horizontal', command=table.xview)
        vertical = ttk.Scrollbar(frame, orient='vertical', command=table.yview)
        table.configure(xscrollcommand=horizontal.set, yscrollcommand=vertical.set)
        table.grid(row=0, column=0, sticky='nsew')
        vertical.grid(row=0, column=1, sticky='ns')
        horizontal.grid(row=1, column=0, sticky='ew')
        headings = dict(shot_id='Test', training_tests='Training tests',
                        full_rmse_v='Full RMSE (V)', full_r2='Full R²',
                        prompt_rmse_v='Prompt RMSE (V)', prompt_r2='Prompt R²')
        for key, label, unit in ridge.PULSE_METRICS:
            headings[f'rmse_{key}'] = f'{label} · RMSE ({unit})'
        for column in columns:
            title = headings[column]
            table.heading(column, text=title)
            table.column(column, width=max(150 if column == 'shot_id' else 100, len(title)*8), minwidth=90,
                         stretch=False, anchor='e')
        table.tag_configure('averages', background='#e1e9f2', font=('Segoe UI', 10, 'bold'))
        for row in ridge.stats_with_totals(data).to_dict('records'):
            summary = row['shot_id'] == 'Total averages'
            table.insert('', 'end', tags=('averages',) if summary else (), values=[
                str(row[column]) if column == 'shot_id' else
                str(int(row[column])) if column == 'training_tests' and not summary else
                (f'{row[column]:.3f}' if isfinite(row[column]) else 'N/A')
                for column in columns])
        return window

    def save_all_stats(self, data=None, parent=None):
        data = self.batch_stats if data is None else data
        if data is None:
            return
        parent = self.root if parent is None else parent
        path = filedialog.asksaveasfilename(parent=parent, title='Download all waveform stats',
                                          defaultextension='.csv', filetypes=[('CSV files', '*.csv')],
                                          initialfile='all_tests_waveform_stats.csv')
        if path:
            try:
                ridge.stats_with_totals(data).to_csv(path, index=False, na_rep='N/A', encoding='utf-8-sig')
                self.status.set(f'Saved waveform stats for {len(data)} tests.')
            except OSError as error:
                messagebox.showerror('Unable to save waveform stats', str(error), parent=parent)

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
