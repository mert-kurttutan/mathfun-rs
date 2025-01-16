
macro_rules! impl_binary {
    ($name:ident, $fn:ident, $ta:ty, $tb:ty) => {
        pub(crate) unsafe fn $name(n: i32, a: *const $ta, b: *const $ta, y: *mut $tb) {
            assert!(n > 0);
            let fn_sequantial = *$fn;
            let num_per_thread_min = 1000000;
            let num_thread_max = 8;
            let n_per_thread = (n + num_thread_max - 1) / num_thread_max;
            let num_thread = if n_per_thread >= num_per_thread_min {
                num_thread_max
            } else {
                (n + num_per_thread_min - 1) / num_per_thread_min
            };
            let num_per_thread = (n + num_thread - 1) / num_thread;
            let a_usize = a as usize;
            let b_usize = b as usize;
            let y_usize = y as usize;
            let x = std::thread::spawn(move || {
                for i in 1..num_thread {
                    let a = a_usize as *const $ta;
                    let b = b_usize as *const $ta;
                    let y = y_usize as *mut $tb;
                    let n_cur = num_per_thread.min(n - i * num_per_thread);
                    fn_sequantial(n_cur, a.offset((i * num_per_thread) as isize), b.offset((i * num_per_thread) as isize), y.offset((i * num_per_thread) as isize));
                }
            });
            fn_sequantial(n.min(num_per_thread), a, b, y);
            x.join().unwrap();
        }
    };
}

