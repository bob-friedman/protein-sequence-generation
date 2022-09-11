use Digest::MD5 qw(md5_hex);

@digests=();

$file1=shift @ARGV;
open(fh1,"$file1");

while($line=<fh1>){
	chomp $line;
	if ($line =~ /^\s*$/){
		next;
	}elsif($line =~ /^\s*#/){
		next;
	}elsif($line =~ /^>/){
		$line =~ s/>//g;
		if($switch==1){
			$i++;
			$digest = md5_hex($sequence);
			$equivalent=0;
			foreach $list(@digests){
				if($list =~ m/^$digest$/){
					# print "$i\t$list\t$digest\n";
					$equivalent=1;
				}
			}
			if($equivalent==0){
				# print ">$i|$digest|$name\n$sequence\n";
				print ">$name\n$sequence\n";
			}
		}
		push(@temp,$digest);
		%seen = ();
		@digests = grep { ! $seen{ $_ }++ } @temp; # remove duplicates
		@temp=@digests;
		$switch=1;
		$name=$line;
		$sequence='';
	 	next;
	}else{
		$sequence .= $line;
	}
	$sequence =~ s/\s//g;
}
$digest = md5_hex($sequence);
$line =~ s/>//g;

$equivalent=0;
foreach $list(@digests){
	if($list  =~ m/^$digest$/){
		$equivalent=1;
	}
}
if($equivalent==0){
	# print ">$i|$digest|$name\n$sequence\n";
	print ">$name\n$sequence\n";
}
