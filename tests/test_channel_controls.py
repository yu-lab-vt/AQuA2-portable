import tkinter as tk
from tkinter import ttk
import unittest
from pathlib import Path
from aqua2.viewer_dialogs import CFUForm,NewAnalysisDialog


class ChannelControls(unittest.TestCase):
    def setUp(self):
        from aqua2.tk_runtime import initialize
        self.root=tk.Tk.__new__(tk.Tk);initialize(self.root);self.root.withdraw()

    def tearDown(self):self.root.destroy()

    def test_visual_channel_selector_reappears_only_for_dual_data(self):
        from types import SimpleNamespace
        from aqua2.viewer import Workbench
        panel=ttk.Frame(self.root);channel=tk.StringVar(value='channel2')
        combo=ttk.Combobox(panel,textvariable=channel);scope=ttk.Combobox(panel);scope.pack()
        window=SimpleNamespace(channel=channel,channel_combo=combo,scope_combo=scope)
        Workbench.update_channels(window,['channel1'])
        self.assertEqual(channel.get(),'channel1');self.assertFalse(combo.pack_info() if combo.winfo_manager() else {})
        Workbench.update_channels(window,['channel1','channel2'])
        self.assertEqual(combo.winfo_manager(),'pack');self.assertEqual(tuple(combo['values']),('channel1','channel2'))
        channel.set('channel2');Workbench.update_channels(window,['channel1'])
        self.assertEqual(channel.get(),'channel1');self.assertEqual(combo.winfo_manager(),'')

    def test_cfu_fields_follow_actual_channels_and_ignore_hidden_values(self):
        panel=ttk.Frame(self.root);form=CFUForm(panel)
        self.assertTrue(form.channel2_widgets)
        self.assertTrue(all(not widget.grid_info() for widget in form.channel2_widgets))
        form.values['overlapThr2'].set('invalid hidden edit')
        self.assertEqual(form.get()['cfuDetect']['overlapThr2'],.5)
        form.set_dual_channel(True)
        self.assertTrue(all(widget.grid_info() for widget in form.channel2_widgets))
        with self.assertRaises(ValueError):form.get()
        form.values['overlapThr2'].set('.7')
        self.assertEqual(form.get()['cfuDetect']['overlapThr2'],.7)
        form.set_dual_channel(False)
        self.assertTrue(all(not widget.grid_info() for widget in form.channel2_widgets))

    def test_import_uses_format_detection_and_has_no_parameter_or_channel_toggle(self):
        jobs=[];dialog=NewAnalysisDialog(self.root,jobs.append);dialog.withdraw()
        self.assertEqual(set(dialog.vars),{'source','frames','regions','landmarks'})
        self.assertFalse(hasattr(dialog,'dual_channel'));self.assertFalse(hasattr(dialog,'saved'))
        dialog.vars['source'].set(str(Path(__file__).resolve()));dialog.submit()
        self.assertTrue(jobs[0]['config']['auto_channels']);self.assertNotIn('channel2',jobs[0]['config'])

    def test_cfu_merge_display_is_normalized_and_preserves_default_algorithm_value(self):
        form=CFUForm(ttk.Frame(self.root),dual_channel=True)
        self.assertEqual(form.values['minimumCurveCorrelation'].get(),'0.5')
        self.assertEqual(form.get()['cfuDetect']['minimumCurveCorrelation'],.925)
        for shown,actual in (('0',.85),('1',1.),('.5',.925)):
            form.values['minimumCurveCorrelation'].set(shown);form.values['minimumCurveCorrelation2'].set(shown)
            self.assertEqual(form.get()['cfuDetect']['minimumCurveCorrelation'],actual)
            self.assertEqual(form.get()['cfuDetect']['minimumCurveCorrelation2'],actual)
        form.values['minimumCurveCorrelation'].set('1.1')
        with self.assertRaises(ValueError):form.get()
