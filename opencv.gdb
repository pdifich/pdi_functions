# /* vim: set filetype=gdb:*/
##########################################
#                                        #
# OpenCV GDB evaluators/views/utilities  #
#                                        #
##########################################

define pmat
	if $argc == 0
		help pmat
	else
		set $mat = $arg0
		set $rows = $mat.rows
		set $cols = $mat.cols
		set $size = $rows * $cols

		set $i = 0
		while $i < $rows
			output *(($arg1 *)$arg0.data+$i*$rows)@$cols
			echo '\n'
			set $i++
		end
	end
end
document pmat
	Prints cv::Mat data in a formatted way
	Syntax: pmat <Mat> <type>
	Note: type must be the correct type
end 

define pmatuc
	if $argc == 0
		help pmat
	else
		set $mat = $arg0
		set $rows = $mat.rows
		set $cols = $mat.cols
		set $size = $rows * $cols

		set $i = 0
		while $i < $rows
			output /u *((uchar *)$arg0.data+$i*$rows)@$cols
			echo '\n'
			set $i++
		end
	end
end
document pmatuc
	Prints a cv::Mat of type unsigned char
	Syntax: pmatuc <Mat>
end 
