import gc,os,threading,unittest
from multiprocessing import Pipe,shared_memory
import numpy as np
from aqua2.ipc import send_value,receive_value
from aqua2.shared_arrays import describe
from aqua2.io import MatStructArray
from aqua2.reconstruction import Reconstruction


def _probe_worker(job,connection,has_input):
    value=receive_value(connection,read_only=True)
    connection.send({'type':'progress','stage':'attached'})
    if job['operation']=='wait':connection.recv()
    elif job['operation']=='fail':
        try:value['movies']['channel1']['working'][0,0,0,0]=99
        except ValueError:connection.send({'type':'error','error':'shared input protected'})
    else:
        connection.send({'type':'result','project_path':None})
        stats=send_value(connection,value,shared=True)
        connection.send({'type':'transport','direction':'output',**stats})
    connection.close()


@unittest.skipUnless(os.name=='nt','Windows named shared-array transport')
class SharedTransportTests(unittest.TestCase):
    def transfer(self,value,read_only=False):
        writer,reader=Pipe();errors=[];stats=[]
        def send():
            try:stats.append(send_value(writer,value,shared=True))
            except BaseException as error:errors.append(error)
            finally:writer.close()
        thread=threading.Thread(target=send);thread.start()
        try:result=receive_value(reader,read_only=read_only)
        finally:reader.close();thread.join(10)
        self.assertFalse(thread.is_alive());self.assertFalse(errors)
        return result,stats[0]

    def test_large_arrays_cross_once_then_only_descriptors(self):
        for order in ('F','C'):
            source=np.arange(400*600,dtype=np.float32).reshape(400,600,order=order)
            first,stats=self.transfer(source)
            np.testing.assert_array_equal(source,first)
            self.assertIs(type(first),np.ndarray)
            self.assertEqual(stats['copied_bytes'],source.nbytes)
            self.assertEqual(stats['pipe_bytes'],0)
            second,stats=self.transfer(first,read_only=True)
            self.assertEqual(stats['copied_bytes'],0)
            self.assertEqual(stats['reused_bytes'],source.nbytes)
            self.assertTrue(np.shares_memory(first,second))
            with self.assertRaises(ValueError):second[0,0]=10
            np.testing.assert_array_equal(source,first)

    def test_contiguous_views_reuse_offset_and_survive_original_release(self):
        source,_=self.transfer(np.arange(400000,dtype=np.int64))
        view=source[2000:80000];name=describe(view).name
        result,stats=self.transfer(view)
        self.assertEqual(describe(result).offset,2000*8)
        self.assertEqual(stats['copied_bytes'],0)
        del source,view;gc.collect()
        np.testing.assert_array_equal(result,np.arange(2000,80000,dtype=np.int64))
        probe=shared_memory.SharedMemory(name=name);probe.close()
        del result;gc.collect()
        with self.assertRaises(FileNotFoundError):shared_memory.SharedMemory(name=name)

    def test_cells_records_and_noncontiguous_arrays_remain_exact(self):
        cells=np.empty((1,1),object).view(MatStructArray);cells.mat_fields=('values',)
        array=np.arange(300000,dtype=np.float32)
        cells[0,0]={'values':array[::2]}
        reconstruction=Reconstruction((1000,1000,1,1),np.arange(100000,dtype=np.int64),np.arange(100000,dtype=np.uint8))
        result,stats=self.transfer({'cells':cells,'reconstruction':reconstruction})
        self.assertEqual(result['cells'].mat_fields,('values',))
        np.testing.assert_array_equal(result['cells'][0,0]['values'],array[::2])
        np.testing.assert_array_equal(result['reconstruction'].pixels,reconstruction.pixels)
        np.testing.assert_array_equal(result['reconstruction'].values,reconstruction.values)
        self.assertGreater(stats['copied_bytes'],0)

    def test_spawn_exit_cancel_and_failed_write_preserve_parent_arrays(self):
        from unittest.mock import patch
        from aqua2.project import new_project
        from aqua2.worker_client import JobClient
        shape=(100,100,1,10);source=np.arange(np.prod(shape),dtype=np.float32).reshape(shape,order='F')
        shared,_=self.transfer(source)
        data=new_project('shared cancellation');data.update(opts={'sz':shape},movies={'channel1':{'working':shared,'stored':shared}})
        client=JobClient()
        for operation in ('echo','fail','wait'):
            with self.subTest(operation=operation):
                events=[]
                with patch('aqua2.worker_client._job_worker',_probe_worker):
                    client.start({'operation':operation,'project_data':data})
                try:
                    while True:
                        event=client.events.get(timeout=30);events.append(event)
                        if operation=='wait' and event.get('stage')=='attached':client.cancel()
                        if event['type']=='exit':break
                finally:client.cancel()
                np.testing.assert_array_equal(shared,source)
                transport=next(e for e in events if e['type']=='transport' and e['direction']=='input')
                self.assertEqual(transport['copied_bytes'],0)
                if operation=='echo':
                    result=next(e['analysis'] for e in events if e['type']=='complete')
                    np.testing.assert_array_equal(result.movie(kind='working'),source)
                    self.assertEqual(describe(result.movie(kind='working')).name,describe(shared).name)
                elif operation=='fail':self.assertTrue(any(e.get('error')=='shared input protected' for e in events))
                else:self.assertTrue(events[-1]['cancelled'])
