phony: clean, deep_clean, testing_clean

clean:
	rm -fv *.log *.aux *.gz *.auxlock *.dep *.dpth *.md5 

deep_clean: clean
	rm -fv *.pdf 

testing_clean: clean
	rm -fv test*[^data].tex test*.pdf 

