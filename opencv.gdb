# /* vim: set filetype=gdb:*/
##########################################
#                                        #
# OpenCV GDB evaluators/views/utilities  #
#                                        #
##########################################


define pmatrix
	if $argc == 0
		help pmatrix
	else
		set $mat = $arg0
		set $rows = $mat.rows
		set $cols = $mat.cols
		set $size = $rows * $cols
	end

	print *(($arg1 *) $arg0.data)@$size
end
document pmatrix
	Prints cv::Mat data
	Syntax: pmatrix <Mat> <type>
	Note: type must be the correct type
end 

define pfmatrix
	if $argc == 0
		help pmatrix
	else
		set $mat = $arg0
		set $rows = $mat.rows
		set $cols = $mat.cols
		set $size = $rows * $cols
	end

	set $i = 0
	while $i < $rows
		print *(($arg1 *)$arg0.data+$i*$rows)@$cols
		set $i++
	end
end

document pfmatrix
	Prints cv::Mat data in a formatted way
	Syntax: pfmatrix <Mat> <type>
	Note: type must be the correct type
end 
